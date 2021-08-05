//===-- IRExecutionUnit.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Log.h"

#include "lldb/../../source/Plugins/Language/CPlusPlus/CPlusPlusLanguage.h"
#include "lldb/../../source/Plugins/ObjectFile/JIT/ObjectFileJIT.h"

using namespace lldb_private;

IRExecutionUnit::IRExecutionUnit(std::unique_ptr<llvm::LLVMContext> &context_up,
                                 std::unique_ptr<llvm::Module> &module_up,
                                 ConstString &name,
                                 const lldb::TargetSP &target_sp,
                                 const SymbolContext &sym_ctx,
                                 std::vector<std::string> &cpu_features)
    : IRMemoryMap(target_sp), m_context_up(context_up.release()),
      m_module_up(module_up.release()), m_module(m_module_up.get()),
      m_cpu_features(cpu_features), m_name(name), m_sym_ctx(sym_ctx),
      m_did_jit(false), m_function_load_addr(LLDB_INVALID_ADDRESS),
      m_function_end_load_addr(LLDB_INVALID_ADDRESS),
      m_reported_allocations(false) {}

lldb::addr_t IRExecutionUnit::WriteNow(const uint8_t *bytes, size_t size,
                                       Status &error) {
  const bool zero_memory = false;
  lldb::addr_t allocation_process_addr =
      Malloc(size, 8, lldb::ePermissionsWritable | lldb::ePermissionsReadable,
             eAllocationPolicyMirror, zero_memory, error);

  if (!error.Success())
    return LLDB_INVALID_ADDRESS;

  WriteMemory(allocation_process_addr, bytes, size, error);

  if (!error.Success()) {
    Status err;
    Free(allocation_process_addr, err);

    return LLDB_INVALID_ADDRESS;
  }

  if (Log *log =
          lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS)) {
    DataBufferHeap my_buffer(size, 0);
    Status err;
    ReadMemory(my_buffer.GetBytes(), allocation_process_addr, size, err);

    if (err.Success()) {
      DataExtractor my_extractor(my_buffer.GetBytes(), my_buffer.GetByteSize(),
                                 lldb::eByteOrderBig, 8);
      my_extractor.PutToLog(log, 0, my_buffer.GetByteSize(),
                            allocation_process_addr, 16,
                            DataExtractor::TypeUInt8);
    }
  }

  return allocation_process_addr;
}

void IRExecutionUnit::FreeNow(lldb::addr_t allocation) {
  if (allocation == LLDB_INVALID_ADDRESS)
    return;

  Status err;

  Free(allocation, err);
}

Status IRExecutionUnit::DisassembleFunction(Stream &stream,
                                            lldb::ProcessSP &process_wp) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  ExecutionContext exe_ctx(process_wp);

  Status ret;

  ret.Clear();

  lldb::addr_t func_local_addr = LLDB_INVALID_ADDRESS;
  lldb::addr_t func_remote_addr = LLDB_INVALID_ADDRESS;

  for (JittedFunction &function : m_jitted_functions) {
    if (function.m_name == m_name) {
      func_local_addr = function.m_local_addr;
      func_remote_addr = function.m_remote_addr;
    }
  }

  if (func_local_addr == LLDB_INVALID_ADDRESS) {
    ret.SetErrorToGenericError();
    ret.SetErrorStringWithFormat("Couldn't find function %s for disassembly",
                                 m_name.AsCString());
    return ret;
  }

  LLDB_LOGF(log,
            "Found function, has local address 0x%" PRIx64
            " and remote address 0x%" PRIx64,
            (uint64_t)func_local_addr, (uint64_t)func_remote_addr);

  std::pair<lldb::addr_t, lldb::addr_t> func_range;

  func_range = GetRemoteRangeForLocal(func_local_addr);

  if (func_range.first == 0 && func_range.second == 0) {
    ret.SetErrorToGenericError();
    ret.SetErrorStringWithFormat("Couldn't find code range for function %s",
                                 m_name.AsCString());
    return ret;
  }

  LLDB_LOGF(log, "Function's code range is [0x%" PRIx64 "+0x%" PRIx64 "]",
            func_range.first, func_range.second);

  Target *target = exe_ctx.GetTargetPtr();
  if (!target) {
    ret.SetErrorToGenericError();
    ret.SetErrorString("Couldn't find the target");
    return ret;
  }

  lldb::DataBufferSP buffer_sp(new DataBufferHeap(func_range.second, 0));

  Process *process = exe_ctx.GetProcessPtr();
  Status err;
  process->ReadMemory(func_remote_addr, buffer_sp->GetBytes(),
                      buffer_sp->GetByteSize(), err);

  if (!err.Success()) {
    ret.SetErrorToGenericError();
    ret.SetErrorStringWithFormat("Couldn't read from process: %s",
                                 err.AsCString("unknown error"));
    return ret;
  }

  ArchSpec arch(target->GetArchitecture());

  const char *plugin_name = nullptr;
  const char *flavor_string = nullptr;
  lldb::DisassemblerSP disassembler_sp =
      Disassembler::FindPlugin(arch, flavor_string, plugin_name);

  if (!disassembler_sp) {
    ret.SetErrorToGenericError();
    ret.SetErrorStringWithFormat(
        "Unable to find disassembler plug-in for %s architecture.",
        arch.GetArchitectureName());
    return ret;
  }

  if (!process) {
    ret.SetErrorToGenericError();
    ret.SetErrorString("Couldn't find the process");
    return ret;
  }

  DataExtractor extractor(buffer_sp, process->GetByteOrder(),
                          target->GetArchitecture().GetAddressByteSize());

  if (log) {
    LLDB_LOGF(log, "Function data has contents:");
    extractor.PutToLog(log, 0, extractor.GetByteSize(), func_remote_addr, 16,
                       DataExtractor::TypeUInt8);
  }

  disassembler_sp->DecodeInstructions(Address(func_remote_addr), extractor, 0,
                                      UINT32_MAX, false, false);

  InstructionList &instruction_list = disassembler_sp->GetInstructionList();
  instruction_list.Dump(&stream, true, true, &exe_ctx);
  return ret;
}

namespace {
struct IRExecDiagnosticHandler : public llvm::DiagnosticHandler {
  Status *err;
  IRExecDiagnosticHandler(Status *err) : err(err) {}
  bool handleDiagnostics(const llvm::DiagnosticInfo &DI) override {
    if (DI.getKind() == llvm::DK_SrcMgr) {
      const auto &DISM = llvm::cast<llvm::DiagnosticInfoSrcMgr>(DI);
      if (err && err->Success()) {
        err->SetErrorToGenericError();
        err->SetErrorStringWithFormat(
            "Inline assembly error: %s",
            DISM.getSMDiag().getMessage().str().c_str());
      }
      return true;
    }

    return false;
  }
};
} // namespace

void IRExecutionUnit::ReportSymbolLookupError(ConstString name) {
  m_failed_lookups.push_back(name);
}

void IRExecutionUnit::GetRunnableInfo(Status &error, lldb::addr_t &func_addr,
                                      lldb::addr_t &func_end) {
  lldb::ProcessSP process_sp(GetProcessWP().lock());

  static std::recursive_mutex s_runnable_info_mutex;

  func_addr = LLDB_INVALID_ADDRESS;
  func_end = LLDB_INVALID_ADDRESS;

  if (!process_sp) {
    error.SetErrorToGenericError();
    error.SetErrorString("Couldn't write the JIT compiled code into the "
                         "process because the process is invalid");
    return;
  }

  if (m_did_jit) {
    func_addr = m_function_load_addr;
    func_end = m_function_end_load_addr;

    return;
  };

  std::lock_guard<std::recursive_mutex> guard(s_runnable_info_mutex);

  m_did_jit = true;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  std::string error_string;

  if (log) {
    std::string s;
    llvm::raw_string_ostream oss(s);

    m_module->print(oss, nullptr);

    oss.flush();

    LLDB_LOGF(log, "Module being sent to JIT: \n%s", s.c_str());
  }

  m_module_up->getContext().setDiagnosticHandler(
      std::make_unique<IRExecDiagnosticHandler>(&error));

  llvm::EngineBuilder builder(std::move(m_module_up));
  llvm::Triple triple(m_module->getTargetTriple());

  builder.setEngineKind(llvm::EngineKind::JIT)
      .setErrorStr(&error_string)
      .setRelocationModel(triple.isOSBinFormatMachO() ? llvm::Reloc::PIC_
                                                      : llvm::Reloc::Static)
      .setMCJITMemoryManager(std::make_unique<MemoryManager>(*this))
      .setOptLevel(llvm::CodeGenOpt::Less);

  llvm::StringRef mArch;
  llvm::StringRef mCPU;
  llvm::SmallVector<std::string, 0> mAttrs;

  for (std::string &feature : m_cpu_features)
    mAttrs.push_back(feature);

  llvm::TargetMachine *target_machine =
      builder.selectTarget(triple, mArch, mCPU, mAttrs);

  m_execution_engine_up.reset(builder.create(target_machine));

  if (!m_execution_engine_up) {
    error.SetErrorToGenericError();
    error.SetErrorStringWithFormat("Couldn't JIT the function: %s",
                                   error_string.c_str());
    return;
  }

  m_strip_underscore =
      (m_execution_engine_up->getDataLayout().getGlobalPrefix() == '_');

  class ObjectDumper : public llvm::ObjectCache {
  public:
    void notifyObjectCompiled(const llvm::Module *module,
                              llvm::MemoryBufferRef object) override {
      int fd = 0;
      llvm::SmallVector<char, 256> result_path;
      std::string object_name_model =
          "jit-object-" + module->getModuleIdentifier() + "-%%%.o";
      (void)llvm::sys::fs::createUniqueFile(object_name_model, fd, result_path);
      llvm::raw_fd_ostream fds(fd, true);
      fds.write(object.getBufferStart(), object.getBufferSize());
    }

    std::unique_ptr<llvm::MemoryBuffer>
    getObject(const llvm::Module *module) override {
      // Return nothing - we're just abusing the object-cache mechanism to dump
      // objects.
      return nullptr;
    }
  };

  if (process_sp->GetTarget().GetEnableSaveObjects()) {
    m_object_cache_up = std::make_unique<ObjectDumper>();
    m_execution_engine_up->setObjectCache(m_object_cache_up.get());
  }

  // Make sure we see all sections, including ones that don't have
  // relocations...
  m_execution_engine_up->setProcessAllSections(true);

  m_execution_engine_up->DisableLazyCompilation();

  for (llvm::Function &function : *m_module) {
    if (function.isDeclaration() || function.hasPrivateLinkage())
      continue;

    const bool external = !function.hasLocalLinkage();

    void *fun_ptr = m_execution_engine_up->getPointerToFunction(&function);

    if (!error.Success()) {
      // We got an error through our callback!
      return;
    }

    if (!fun_ptr) {
      error.SetErrorToGenericError();
      error.SetErrorStringWithFormat(
          "'%s' was in the JITted module but wasn't lowered",
          function.getName().str().c_str());
      return;
    }
    m_jitted_functions.push_back(JittedFunction(
        function.getName().str().c_str(), external, reinterpret_cast<uintptr_t>(fun_ptr)));
  }

  CommitAllocations(process_sp);
  ReportAllocations(*m_execution_engine_up);

  // We have to do this after calling ReportAllocations because for the MCJIT,
  // getGlobalValueAddress will cause the JIT to perform all relocations.  That
  // can only be done once, and has to happen after we do the remapping from
  // local -> remote. That means we don't know the local address of the
  // Variables, but we don't need that for anything, so that's okay.

  std::function<void(llvm::GlobalValue &)> RegisterOneValue = [this](
      llvm::GlobalValue &val) {
    if (val.hasExternalLinkage() && !val.isDeclaration()) {
      uint64_t var_ptr_addr =
          m_execution_engine_up->getGlobalValueAddress(val.getName().str());

      lldb::addr_t remote_addr = GetRemoteAddressForLocal(var_ptr_addr);

      // This is a really unfortunae API that sometimes returns local addresses
      // and sometimes returns remote addresses, based on whether the variable
      // was relocated during ReportAllocations or not.

      if (remote_addr == LLDB_INVALID_ADDRESS) {
        remote_addr = var_ptr_addr;
      }

      if (var_ptr_addr != 0)
        m_jitted_global_variables.push_back(JittedGlobalVariable(
            val.getName().str().c_str(), LLDB_INVALID_ADDRESS, remote_addr));
    }
  };

  for (llvm::GlobalVariable &global_var : m_module->getGlobalList()) {
    RegisterOneValue(global_var);
  }

  for (llvm::GlobalAlias &global_alias : m_module->getAliasList()) {
    RegisterOneValue(global_alias);
  }

  WriteData(process_sp);

  if (m_failed_lookups.size()) {
    StreamString ss;

    ss.PutCString("Couldn't lookup symbols:\n");

    bool emitNewLine = false;

    for (ConstString failed_lookup : m_failed_lookups) {
      if (emitNewLine)
        ss.PutCString("\n");
      emitNewLine = true;
      ss.PutCString("  ");
      ss.PutCString(Mangled(failed_lookup).GetDemangledName().GetStringRef());
    }

    m_failed_lookups.clear();

    error.SetErrorString(ss.GetString());

    return;
  }

  m_function_load_addr = LLDB_INVALID_ADDRESS;
  m_function_end_load_addr = LLDB_INVALID_ADDRESS;

  for (JittedFunction &jitted_function : m_jitted_functions) {
    jitted_function.m_remote_addr =
        GetRemoteAddressForLocal(jitted_function.m_local_addr);

    if (!m_name.IsEmpty() && jitted_function.m_name == m_name) {
      AddrRange func_range =
          GetRemoteRangeForLocal(jitted_function.m_local_addr);
      m_function_end_load_addr = func_range.first + func_range.second;
      m_function_load_addr = jitted_function.m_remote_addr;
    }
  }

  if (log) {
    LLDB_LOGF(log, "Code can be run in the target.");

    StreamString disassembly_stream;

    Status err = DisassembleFunction(disassembly_stream, process_sp);

    if (!err.Success()) {
      LLDB_LOGF(log, "Couldn't disassemble function : %s",
                err.AsCString("unknown error"));
    } else {
      LLDB_LOGF(log, "Function disassembly:\n%s", disassembly_stream.GetData());
    }

    LLDB_LOGF(log, "Sections: ");
    for (AllocationRecord &record : m_records) {
      if (record.m_process_address != LLDB_INVALID_ADDRESS) {
        record.dump(log);

        DataBufferHeap my_buffer(record.m_size, 0);
        Status err;
        ReadMemory(my_buffer.GetBytes(), record.m_process_address,
                   record.m_size, err);

        if (err.Success()) {
          DataExtractor my_extractor(my_buffer.GetBytes(),
                                     my_buffer.GetByteSize(),
                                     lldb::eByteOrderBig, 8);
          my_extractor.PutToLog(log, 0, my_buffer.GetByteSize(),
                                record.m_process_address, 16,
                                DataExtractor::TypeUInt8);
        }
      } else {
        record.dump(log);

        DataExtractor my_extractor((const void *)record.m_host_address,
                                   record.m_size, lldb::eByteOrderBig, 8);
        my_extractor.PutToLog(log, 0, record.m_size, record.m_host_address, 16,
                              DataExtractor::TypeUInt8);
      }
    }
  }

  func_addr = m_function_load_addr;
  func_end = m_function_end_load_addr;

  return;
}

IRExecutionUnit::~IRExecutionUnit() {
  m_module_up.reset();
  m_execution_engine_up.reset();
  m_context_up.reset();
}

IRExecutionUnit::MemoryManager::MemoryManager(IRExecutionUnit &parent)
    : m_default_mm_up(new llvm::SectionMemoryManager()), m_parent(parent) {}

IRExecutionUnit::MemoryManager::~MemoryManager() = default;

lldb::SectionType IRExecutionUnit::GetSectionTypeFromSectionName(
    const llvm::StringRef &name, IRExecutionUnit::AllocationKind alloc_kind) {
  lldb::SectionType sect_type = lldb::eSectionTypeCode;
  switch (alloc_kind) {
  case AllocationKind::Stub:
    sect_type = lldb::eSectionTypeCode;
    break;
  case AllocationKind::Code:
    sect_type = lldb::eSectionTypeCode;
    break;
  case AllocationKind::Data:
    sect_type = lldb::eSectionTypeData;
    break;
  case AllocationKind::Global:
    sect_type = lldb::eSectionTypeData;
    break;
  case AllocationKind::Bytes:
    sect_type = lldb::eSectionTypeOther;
    break;
  }

  if (!name.empty()) {
    if (name.equals("__text") || name.equals(".text"))
      sect_type = lldb::eSectionTypeCode;
    else if (name.equals("__data") || name.equals(".data"))
      sect_type = lldb::eSectionTypeCode;
    else if (name.startswith("__debug_") || name.startswith(".debug_")) {
      const uint32_t name_idx = name[0] == '_' ? 8 : 7;
      llvm::StringRef dwarf_name(name.substr(name_idx));
      switch (dwarf_name[0]) {
      case 'a':
        if (dwarf_name.equals("abbrev"))
          sect_type = lldb::eSectionTypeDWARFDebugAbbrev;
        else if (dwarf_name.equals("aranges"))
          sect_type = lldb::eSectionTypeDWARFDebugAranges;
        else if (dwarf_name.equals("addr"))
          sect_type = lldb::eSectionTypeDWARFDebugAddr;
        break;

      case 'f':
        if (dwarf_name.equals("frame"))
          sect_type = lldb::eSectionTypeDWARFDebugFrame;
        break;

      case 'i':
        if (dwarf_name.equals("info"))
          sect_type = lldb::eSectionTypeDWARFDebugInfo;
        break;

      case 'l':
        if (dwarf_name.equals("line"))
          sect_type = lldb::eSectionTypeDWARFDebugLine;
        else if (dwarf_name.equals("loc"))
          sect_type = lldb::eSectionTypeDWARFDebugLoc;
        else if (dwarf_name.equals("loclists"))
          sect_type = lldb::eSectionTypeDWARFDebugLocLists;
        break;

      case 'm':
        if (dwarf_name.equals("macinfo"))
          sect_type = lldb::eSectionTypeDWARFDebugMacInfo;
        break;

      case 'p':
        if (dwarf_name.equals("pubnames"))
          sect_type = lldb::eSectionTypeDWARFDebugPubNames;
        else if (dwarf_name.equals("pubtypes"))
          sect_type = lldb::eSectionTypeDWARFDebugPubTypes;
        break;

      case 's':
        if (dwarf_name.equals("str"))
          sect_type = lldb::eSectionTypeDWARFDebugStr;
        else if (dwarf_name.equals("str_offsets"))
          sect_type = lldb::eSectionTypeDWARFDebugStrOffsets;
        break;

      case 'r':
        if (dwarf_name.equals("ranges"))
          sect_type = lldb::eSectionTypeDWARFDebugRanges;
        break;

      default:
        break;
      }
    } else if (name.startswith("__apple_") || name.startswith(".apple_"))
      sect_type = lldb::eSectionTypeInvalid;
    else if (name.equals("__objc_imageinfo"))
      sect_type = lldb::eSectionTypeOther;
  }
  return sect_type;
}

uint8_t *IRExecutionUnit::MemoryManager::allocateCodeSection(
    uintptr_t Size, unsigned Alignment, unsigned SectionID,
    llvm::StringRef SectionName) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  uint8_t *return_value = m_default_mm_up->allocateCodeSection(
      Size, Alignment, SectionID, SectionName);

  m_parent.m_records.push_back(AllocationRecord(
      (uintptr_t)return_value,
      lldb::ePermissionsReadable | lldb::ePermissionsExecutable,
      GetSectionTypeFromSectionName(SectionName, AllocationKind::Code), Size,
      Alignment, SectionID, SectionName.str().c_str()));

  LLDB_LOGF(log,
            "IRExecutionUnit::allocateCodeSection(Size=0x%" PRIx64
            ", Alignment=%u, SectionID=%u) = %p",
            (uint64_t)Size, Alignment, SectionID, (void *)return_value);

  if (m_parent.m_reported_allocations) {
    Status err;
    lldb::ProcessSP process_sp =
        m_parent.GetBestExecutionContextScope()->CalculateProcess();

    m_parent.CommitOneAllocation(process_sp, err, m_parent.m_records.back());
  }

  return return_value;
}

uint8_t *IRExecutionUnit::MemoryManager::allocateDataSection(
    uintptr_t Size, unsigned Alignment, unsigned SectionID,
    llvm::StringRef SectionName, bool IsReadOnly) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  uint8_t *return_value = m_default_mm_up->allocateDataSection(
      Size, Alignment, SectionID, SectionName, IsReadOnly);

  uint32_t permissions = lldb::ePermissionsReadable;
  if (!IsReadOnly)
    permissions |= lldb::ePermissionsWritable;
  m_parent.m_records.push_back(AllocationRecord(
      (uintptr_t)return_value, permissions,
      GetSectionTypeFromSectionName(SectionName, AllocationKind::Data), Size,
      Alignment, SectionID, SectionName.str().c_str()));
  LLDB_LOGF(log,
            "IRExecutionUnit::allocateDataSection(Size=0x%" PRIx64
            ", Alignment=%u, SectionID=%u) = %p",
            (uint64_t)Size, Alignment, SectionID, (void *)return_value);

  if (m_parent.m_reported_allocations) {
    Status err;
    lldb::ProcessSP process_sp =
        m_parent.GetBestExecutionContextScope()->CalculateProcess();

    m_parent.CommitOneAllocation(process_sp, err, m_parent.m_records.back());
  }

  return return_value;
}

static ConstString FindBestAlternateMangledName(ConstString demangled,
                                                const SymbolContext &sym_ctx) {
  CPlusPlusLanguage::MethodName cpp_name(demangled);
  std::string scope_qualified_name = cpp_name.GetScopeQualifiedName();

  if (!scope_qualified_name.size())
    return ConstString();

  if (!sym_ctx.module_sp)
    return ConstString();

  lldb_private::SymbolFile *sym_file = sym_ctx.module_sp->GetSymbolFile();
  if (!sym_file)
    return ConstString();

  std::vector<ConstString> alternates;
  sym_file->GetMangledNamesForFunction(scope_qualified_name, alternates);

  std::vector<ConstString> param_and_qual_matches;
  std::vector<ConstString> param_matches;
  for (size_t i = 0; i < alternates.size(); i++) {
    ConstString alternate_mangled_name = alternates[i];
    Mangled mangled(alternate_mangled_name);
    ConstString demangled = mangled.GetDemangledName();

    CPlusPlusLanguage::MethodName alternate_cpp_name(demangled);
    if (!cpp_name.IsValid())
      continue;

    if (alternate_cpp_name.GetArguments() == cpp_name.GetArguments()) {
      if (alternate_cpp_name.GetQualifiers() == cpp_name.GetQualifiers())
        param_and_qual_matches.push_back(alternate_mangled_name);
      else
        param_matches.push_back(alternate_mangled_name);
    }
  }

  if (param_and_qual_matches.size())
    return param_and_qual_matches[0]; // It is assumed that there will be only
                                      // one!
  else if (param_matches.size())
    return param_matches[0]; // Return one of them as a best match
  else
    return ConstString();
}

struct IRExecutionUnit::SearchSpec {
  ConstString name;
  lldb::FunctionNameType mask;

  SearchSpec(ConstString n,
             lldb::FunctionNameType m = lldb::eFunctionNameTypeFull)
      : name(n), mask(m) {}
};

void IRExecutionUnit::CollectCandidateCNames(
    std::vector<IRExecutionUnit::SearchSpec> &C_specs,
    ConstString name) {
  if (m_strip_underscore && name.AsCString()[0] == '_')
    C_specs.insert(C_specs.begin(), ConstString(&name.AsCString()[1]));
  C_specs.push_back(SearchSpec(name));
}

void IRExecutionUnit::CollectCandidateCPlusPlusNames(
    std::vector<IRExecutionUnit::SearchSpec> &CPP_specs,
    const std::vector<SearchSpec> &C_specs, const SymbolContext &sc) {
  for (const SearchSpec &C_spec : C_specs) {
    ConstString name = C_spec.name;

    if (CPlusPlusLanguage::IsCPPMangledName(name.GetCString())) {
      Mangled mangled(name);
      ConstString demangled = mangled.GetDemangledName();

      if (demangled) {
        ConstString best_alternate_mangled_name =
            FindBestAlternateMangledName(demangled, sc);

        if (best_alternate_mangled_name) {
          CPP_specs.push_back(best_alternate_mangled_name);
        }
      }
    }

    std::set<ConstString> alternates;
    CPlusPlusLanguage::FindAlternateFunctionManglings(name, alternates);
    CPP_specs.insert(CPP_specs.end(), alternates.begin(), alternates.end());
  }
}

void IRExecutionUnit::CollectFallbackNames(
    std::vector<SearchSpec> &fallback_specs,
    const std::vector<SearchSpec> &C_specs) {
  // As a last-ditch fallback, try the base name for C++ names.  It's terrible,
  // but the DWARF doesn't always encode "extern C" correctly.

  for (const SearchSpec &C_spec : C_specs) {
    ConstString name = C_spec.name;

    if (!CPlusPlusLanguage::IsCPPMangledName(name.GetCString()))
      continue;

    Mangled mangled_name(name);
    ConstString demangled_name = mangled_name.GetDemangledName();
    if (demangled_name.IsEmpty())
      continue;

    const char *demangled_cstr = demangled_name.AsCString();
    const char *lparen_loc = strchr(demangled_cstr, '(');
    if (!lparen_loc)
      continue;

    llvm::StringRef base_name(demangled_cstr,
                              lparen_loc - demangled_cstr);
    fallback_specs.push_back(ConstString(base_name));
  }
}

lldb::addr_t IRExecutionUnit::FindInSymbols(
    const std::vector<IRExecutionUnit::SearchSpec> &specs,
    const lldb_private::SymbolContext &sc,
    bool &symbol_was_missing_weak) {
  symbol_was_missing_weak = false;
  Target *target = sc.target_sp.get();

  if (!target) {
    // we shouldn't be doing any symbol lookup at all without a target
    return LLDB_INVALID_ADDRESS;
  }

  for (const SearchSpec &spec : specs) {
    SymbolContextList sc_list;

    lldb::addr_t best_internal_load_address = LLDB_INVALID_ADDRESS;

    std::function<bool(lldb::addr_t &, SymbolContextList &,
                       const lldb_private::SymbolContext &)>
        get_external_load_address = [&best_internal_load_address, target,
                                     &symbol_was_missing_weak](
            lldb::addr_t &load_address, SymbolContextList &sc_list,
            const lldb_private::SymbolContext &sc) -> lldb::addr_t {
      load_address = LLDB_INVALID_ADDRESS;

      if (sc_list.GetSize() == 0)
        return false;

      // missing_weak_symbol will be true only if we found only weak undefined 
      // references to this symbol.
      symbol_was_missing_weak = true;      
      for (auto candidate_sc : sc_list.SymbolContexts()) {        
        // Only symbols can be weak undefined:
        if (!candidate_sc.symbol)
          symbol_was_missing_weak = false;
        else if (candidate_sc.symbol->GetType() != lldb::eSymbolTypeUndefined
                  || !candidate_sc.symbol->IsWeak())
          symbol_was_missing_weak = false;
        
        const bool is_external =
            (candidate_sc.function) ||
            (candidate_sc.symbol && candidate_sc.symbol->IsExternal());
        if (candidate_sc.symbol) {
          load_address = candidate_sc.symbol->ResolveCallableAddress(*target);

          if (load_address == LLDB_INVALID_ADDRESS) {
            if (target->GetProcessSP())
              load_address =
                  candidate_sc.symbol->GetAddress().GetLoadAddress(target);
            else
              load_address = candidate_sc.symbol->GetAddress().GetFileAddress();
          }
        }

        if (load_address == LLDB_INVALID_ADDRESS && candidate_sc.function) {
          if (target->GetProcessSP())
            load_address = candidate_sc.function->GetAddressRange()
                               .GetBaseAddress()
                               .GetLoadAddress(target);
          else
            load_address = candidate_sc.function->GetAddressRange()
                               .GetBaseAddress()
                               .GetFileAddress();
        }

        if (load_address != LLDB_INVALID_ADDRESS) {
          if (is_external) {
            return true;
          } else if (best_internal_load_address == LLDB_INVALID_ADDRESS) {
            best_internal_load_address = load_address;
            load_address = LLDB_INVALID_ADDRESS;
          }
        }
      }

      // You test the address of a weak symbol against NULL to see if it is
      // present.  So we should return 0 for a missing weak symbol.
      if (symbol_was_missing_weak) {
        load_address = 0;
        return true;
      }
      
      return false;
    };

    ModuleFunctionSearchOptions function_options;
    function_options.include_symbols = true;
    function_options.include_inlines = false;

    if (sc.module_sp) {
      sc.module_sp->FindFunctions(spec.name, CompilerDeclContext(), spec.mask,
                                  function_options, sc_list);
    }

    lldb::addr_t load_address = LLDB_INVALID_ADDRESS;

    if (get_external_load_address(load_address, sc_list, sc)) {
      return load_address;
    } else {
      sc_list.Clear();
    }

    if (sc_list.GetSize() == 0 && sc.target_sp) {
      sc.target_sp->GetImages().FindFunctions(spec.name, spec.mask,
                                              function_options, sc_list);
    }

    if (get_external_load_address(load_address, sc_list, sc)) {
      return load_address;
    } else {
      sc_list.Clear();
    }

    if (sc_list.GetSize() == 0 && sc.target_sp) {
      sc.target_sp->GetImages().FindSymbolsWithNameAndType(
          spec.name, lldb::eSymbolTypeAny, sc_list);
    }

    if (get_external_load_address(load_address, sc_list, sc)) {
      return load_address;
    }
    // if there are any searches we try after this, add an sc_list.Clear() in
    // an "else" clause here

    if (best_internal_load_address != LLDB_INVALID_ADDRESS) {
      return best_internal_load_address;
    }
  }

  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t
IRExecutionUnit::FindInRuntimes(const std::vector<SearchSpec> &specs,
                                const lldb_private::SymbolContext &sc) {
  lldb::TargetSP target_sp = sc.target_sp;

  if (!target_sp) {
    return LLDB_INVALID_ADDRESS;
  }

  lldb::ProcessSP process_sp = sc.target_sp->GetProcessSP();

  if (!process_sp) {
    return LLDB_INVALID_ADDRESS;
  }

  for (const SearchSpec &spec : specs) {
    for (LanguageRuntime *runtime : process_sp->GetLanguageRuntimes()) {
      lldb::addr_t symbol_load_addr = runtime->LookupRuntimeSymbol(spec.name);

      if (symbol_load_addr != LLDB_INVALID_ADDRESS)
        return symbol_load_addr;
    }
  }

  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t IRExecutionUnit::FindInUserDefinedSymbols(
    const std::vector<SearchSpec> &specs,
    const lldb_private::SymbolContext &sc) {
  lldb::TargetSP target_sp = sc.target_sp;

  for (const SearchSpec &spec : specs) {
    lldb::addr_t symbol_load_addr = target_sp->GetPersistentSymbol(spec.name);

    if (symbol_load_addr != LLDB_INVALID_ADDRESS)
      return symbol_load_addr;
  }

  return LLDB_INVALID_ADDRESS;
}

lldb::addr_t
IRExecutionUnit::FindSymbol(lldb_private::ConstString name, bool &missing_weak) {
  std::vector<SearchSpec> candidate_C_names;
  std::vector<SearchSpec> candidate_CPlusPlus_names;

  CollectCandidateCNames(candidate_C_names, name);
  
  lldb::addr_t ret = FindInSymbols(candidate_C_names, m_sym_ctx, missing_weak);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;
  
  // If we find the symbol in runtimes or user defined symbols it can't be 
  // a missing weak symbol.
  missing_weak = false;
  ret = FindInRuntimes(candidate_C_names, m_sym_ctx);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;

  ret = FindInUserDefinedSymbols(candidate_C_names, m_sym_ctx);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;

  CollectCandidateCPlusPlusNames(candidate_CPlusPlus_names, candidate_C_names,
                                 m_sym_ctx);
  ret = FindInSymbols(candidate_CPlusPlus_names, m_sym_ctx, missing_weak);
  if (ret != LLDB_INVALID_ADDRESS)
    return ret;

  std::vector<SearchSpec> candidate_fallback_names;

  CollectFallbackNames(candidate_fallback_names, candidate_C_names);
  ret = FindInSymbols(candidate_fallback_names, m_sym_ctx, missing_weak);

  return ret;
}

void IRExecutionUnit::GetStaticInitializers(
    std::vector<lldb::addr_t> &static_initializers) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  llvm::GlobalVariable *global_ctors =
      m_module->getNamedGlobal("llvm.global_ctors");
  if (!global_ctors) {
    LLDB_LOG(log, "Couldn't find llvm.global_ctors.");
    return;
  }
  auto *ctor_array =
      llvm::dyn_cast<llvm::ConstantArray>(global_ctors->getInitializer());
  if (!ctor_array) {
    LLDB_LOG(log, "llvm.global_ctors not a ConstantArray.");
    return;
  }

  for (llvm::Use &ctor_use : ctor_array->operands()) {
    auto *ctor_struct = llvm::dyn_cast<llvm::ConstantStruct>(ctor_use);
    if (!ctor_struct)
      continue;
    // this is standardized
    lldbassert(ctor_struct->getNumOperands() == 3);
    auto *ctor_function =
        llvm::dyn_cast<llvm::Function>(ctor_struct->getOperand(1));
    if (!ctor_function) {
      LLDB_LOG(log, "global_ctor doesn't contain an llvm::Function");
      continue;
    }

    ConstString ctor_function_name(ctor_function->getName().str());
    LLDB_LOG(log, "Looking for callable jitted function with name {0}.",
             ctor_function_name);

    for (JittedFunction &jitted_function : m_jitted_functions) {
      if (ctor_function_name != jitted_function.m_name)
        continue;
      if (jitted_function.m_remote_addr == LLDB_INVALID_ADDRESS) {
        LLDB_LOG(log, "Found jitted function with invalid address.");
        continue;
      }
      static_initializers.push_back(jitted_function.m_remote_addr);
      LLDB_LOG(log, "Calling function at address {0:x}.",
               jitted_function.m_remote_addr);
      break;
    }
  }
}

llvm::JITSymbol 
IRExecutionUnit::MemoryManager::findSymbol(const std::string &Name) {
    bool missing_weak = false;
    uint64_t addr = GetSymbolAddressAndPresence(Name, missing_weak);
    // This is a weak symbol:
    if (missing_weak) 
      return llvm::JITSymbol(addr, 
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Weak);
    else
      return llvm::JITSymbol(addr, llvm::JITSymbolFlags::Exported);
}

uint64_t
IRExecutionUnit::MemoryManager::getSymbolAddress(const std::string &Name) {
  bool missing_weak = false;
  return GetSymbolAddressAndPresence(Name, missing_weak);
}

uint64_t 
IRExecutionUnit::MemoryManager::GetSymbolAddressAndPresence(
    const std::string &Name, bool &missing_weak) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  ConstString name_cs(Name.c_str());

  lldb::addr_t ret = m_parent.FindSymbol(name_cs, missing_weak);

  if (ret == LLDB_INVALID_ADDRESS) {
    LLDB_LOGF(log,
              "IRExecutionUnit::getSymbolAddress(Name=\"%s\") = <not found>",
              Name.c_str());

    m_parent.ReportSymbolLookupError(name_cs);
    return 0;
  } else {
    LLDB_LOGF(log, "IRExecutionUnit::getSymbolAddress(Name=\"%s\") = %" PRIx64,
              Name.c_str(), ret);
    return ret;
  }
}

void *IRExecutionUnit::MemoryManager::getPointerToNamedFunction(
    const std::string &Name, bool AbortOnFailure) {
  return (void *)getSymbolAddress(Name);
}

lldb::addr_t
IRExecutionUnit::GetRemoteAddressForLocal(lldb::addr_t local_address) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

  for (AllocationRecord &record : m_records) {
    if (local_address >= record.m_host_address &&
        local_address < record.m_host_address + record.m_size) {
      if (record.m_process_address == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;

      lldb::addr_t ret =
          record.m_process_address + (local_address - record.m_host_address);

      LLDB_LOGF(log,
                "IRExecutionUnit::GetRemoteAddressForLocal() found 0x%" PRIx64
                " in [0x%" PRIx64 "..0x%" PRIx64 "], and returned 0x%" PRIx64
                " from [0x%" PRIx64 "..0x%" PRIx64 "].",
                local_address, (uint64_t)record.m_host_address,
                (uint64_t)record.m_host_address + (uint64_t)record.m_size, ret,
                record.m_process_address,
                record.m_process_address + record.m_size);

      return ret;
    }
  }

  return LLDB_INVALID_ADDRESS;
}

IRExecutionUnit::AddrRange
IRExecutionUnit::GetRemoteRangeForLocal(lldb::addr_t local_address) {
  for (AllocationRecord &record : m_records) {
    if (local_address >= record.m_host_address &&
        local_address < record.m_host_address + record.m_size) {
      if (record.m_process_address == LLDB_INVALID_ADDRESS)
        return AddrRange(0, 0);

      return AddrRange(record.m_process_address, record.m_size);
    }
  }

  return AddrRange(0, 0);
}

bool IRExecutionUnit::CommitOneAllocation(lldb::ProcessSP &process_sp,
                                          Status &error,
                                          AllocationRecord &record) {
  if (record.m_process_address != LLDB_INVALID_ADDRESS) {
    return true;
  }

  switch (record.m_sect_type) {
  case lldb::eSectionTypeInvalid:
  case lldb::eSectionTypeDWARFDebugAbbrev:
  case lldb::eSectionTypeDWARFDebugAddr:
  case lldb::eSectionTypeDWARFDebugAranges:
  case lldb::eSectionTypeDWARFDebugCuIndex:
  case lldb::eSectionTypeDWARFDebugFrame:
  case lldb::eSectionTypeDWARFDebugInfo:
  case lldb::eSectionTypeDWARFDebugLine:
  case lldb::eSectionTypeDWARFDebugLoc:
  case lldb::eSectionTypeDWARFDebugLocLists:
  case lldb::eSectionTypeDWARFDebugMacInfo:
  case lldb::eSectionTypeDWARFDebugPubNames:
  case lldb::eSectionTypeDWARFDebugPubTypes:
  case lldb::eSectionTypeDWARFDebugRanges:
  case lldb::eSectionTypeDWARFDebugStr:
  case lldb::eSectionTypeDWARFDebugStrOffsets:
  case lldb::eSectionTypeDWARFAppleNames:
  case lldb::eSectionTypeDWARFAppleTypes:
  case lldb::eSectionTypeDWARFAppleNamespaces:
  case lldb::eSectionTypeDWARFAppleObjC:
  case lldb::eSectionTypeDWARFGNUDebugAltLink:
    error.Clear();
    break;
  default:
    const bool zero_memory = false;
    record.m_process_address =
        Malloc(record.m_size, record.m_alignment, record.m_permissions,
               eAllocationPolicyProcessOnly, zero_memory, error);
    break;
  }

  return error.Success();
}

bool IRExecutionUnit::CommitAllocations(lldb::ProcessSP &process_sp) {
  bool ret = true;

  lldb_private::Status err;

  for (AllocationRecord &record : m_records) {
    ret = CommitOneAllocation(process_sp, err, record);

    if (!ret) {
      break;
    }
  }

  if (!ret) {
    for (AllocationRecord &record : m_records) {
      if (record.m_process_address != LLDB_INVALID_ADDRESS) {
        Free(record.m_process_address, err);
        record.m_process_address = LLDB_INVALID_ADDRESS;
      }
    }
  }

  return ret;
}

void IRExecutionUnit::ReportAllocations(llvm::ExecutionEngine &engine) {
  m_reported_allocations = true;

  for (AllocationRecord &record : m_records) {
    if (record.m_process_address == LLDB_INVALID_ADDRESS)
      continue;

    if (record.m_section_id == eSectionIDInvalid)
      continue;

    engine.mapSectionAddress((void *)record.m_host_address,
                             record.m_process_address);
  }

  // Trigger re-application of relocations.
  engine.finalizeObject();
}

bool IRExecutionUnit::WriteData(lldb::ProcessSP &process_sp) {
  bool wrote_something = false;
  for (AllocationRecord &record : m_records) {
    if (record.m_process_address != LLDB_INVALID_ADDRESS) {
      lldb_private::Status err;
      WriteMemory(record.m_process_address, (uint8_t *)record.m_host_address,
                  record.m_size, err);
      if (err.Success())
        wrote_something = true;
    }
  }
  return wrote_something;
}

void IRExecutionUnit::AllocationRecord::dump(Log *log) {
  if (!log)
    return;

  LLDB_LOGF(log,
            "[0x%llx+0x%llx]->0x%llx (alignment %d, section ID %d, name %s)",
            (unsigned long long)m_host_address, (unsigned long long)m_size,
            (unsigned long long)m_process_address, (unsigned)m_alignment,
            (unsigned)m_section_id, m_name.c_str());
}

lldb::ByteOrder IRExecutionUnit::GetByteOrder() const {
  ExecutionContext exe_ctx(GetBestExecutionContextScope());
  return exe_ctx.GetByteOrder();
}

uint32_t IRExecutionUnit::GetAddressByteSize() const {
  ExecutionContext exe_ctx(GetBestExecutionContextScope());
  return exe_ctx.GetAddressByteSize();
}

void IRExecutionUnit::PopulateSymtab(lldb_private::ObjectFile *obj_file,
                                     lldb_private::Symtab &symtab) {
  // No symbols yet...
}

void IRExecutionUnit::PopulateSectionList(
    lldb_private::ObjectFile *obj_file,
    lldb_private::SectionList &section_list) {
  for (AllocationRecord &record : m_records) {
    if (record.m_size > 0) {
      lldb::SectionSP section_sp(new lldb_private::Section(
          obj_file->GetModule(), obj_file, record.m_section_id,
          ConstString(record.m_name), record.m_sect_type,
          record.m_process_address, record.m_size,
          record.m_host_address, // file_offset (which is the host address for
                                 // the data)
          record.m_size,         // file_size
          0,
          record.m_permissions)); // flags
      section_list.AddSection(section_sp);
    }
  }
}

ArchSpec IRExecutionUnit::GetArchitecture() {
  ExecutionContext exe_ctx(GetBestExecutionContextScope());
  if(Target *target = exe_ctx.GetTargetPtr())
    return target->GetArchitecture();
  return ArchSpec();
}

lldb::ModuleSP IRExecutionUnit::GetJITModule() {
  ExecutionContext exe_ctx(GetBestExecutionContextScope());
  Target *target = exe_ctx.GetTargetPtr();
  if (!target)
    return nullptr;

  auto Delegate = std::static_pointer_cast<lldb_private::ObjectFileJITDelegate>(
      shared_from_this());

  lldb::ModuleSP jit_module_sp =
      lldb_private::Module::CreateModuleFromObjectFile<ObjectFileJIT>(Delegate);
  if (!jit_module_sp)
    return nullptr;

  bool changed = false;
  jit_module_sp->SetLoadAddress(*target, 0, true, changed);
  return jit_module_sp;
}
