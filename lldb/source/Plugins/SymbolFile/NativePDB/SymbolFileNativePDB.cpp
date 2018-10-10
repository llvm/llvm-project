//===-- SymbolFileNativePDB.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFileNativePDB.h"

#include "clang/Lex/Lexer.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolVendor.h"

#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/DebugLinesSubsection.h"
#include "llvm/DebugInfo/CodeView/RecordName.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/SymbolStream.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#include "PdbSymUid.h"
#include "PdbUtil.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::npdb;
using namespace llvm::codeview;
using namespace llvm::pdb;

static lldb::LanguageType TranslateLanguage(PDB_Lang lang) {
  switch (lang) {
  case PDB_Lang::Cpp:
    return lldb::LanguageType::eLanguageTypeC_plus_plus;
  case PDB_Lang::C:
    return lldb::LanguageType::eLanguageTypeC;
  default:
    return lldb::LanguageType::eLanguageTypeUnknown;
  }
}

static std::unique_ptr<PDBFile> loadPDBFile(std::string PdbPath,
                                            llvm::BumpPtrAllocator &Allocator) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ErrorOrBuffer =
      llvm::MemoryBuffer::getFile(PdbPath, /*FileSize=*/-1,
                                  /*RequiresNullTerminator=*/false);
  if (!ErrorOrBuffer)
    return nullptr;
  std::unique_ptr<llvm::MemoryBuffer> Buffer = std::move(*ErrorOrBuffer);

  llvm::StringRef Path = Buffer->getBufferIdentifier();
  auto Stream = llvm::make_unique<llvm::MemoryBufferByteStream>(
      std::move(Buffer), llvm::support::little);

  auto File = llvm::make_unique<PDBFile>(Path, std::move(Stream), Allocator);
  if (auto EC = File->parseFileHeaders())
    return nullptr;
  if (auto EC = File->parseStreamData())
    return nullptr;

  return std::move(File);
}

static std::unique_ptr<PDBFile>
loadMatchingPDBFile(std::string exe_path, llvm::BumpPtrAllocator &allocator) {
  // Try to find a matching PDB for an EXE.
  using namespace llvm::object;
  auto expected_binary = createBinary(exe_path);

  // If the file isn't a PE/COFF executable, fail.
  if (!expected_binary) {
    llvm::consumeError(expected_binary.takeError());
    return nullptr;
  }
  OwningBinary<Binary> binary = std::move(*expected_binary);

  auto *obj = llvm::dyn_cast<llvm::object::COFFObjectFile>(binary.getBinary());
  if (!obj)
    return nullptr;
  const llvm::codeview::DebugInfo *pdb_info = nullptr;

  // If it doesn't have a debug directory, fail.
  llvm::StringRef pdb_file;
  auto ec = obj->getDebugPDBInfo(pdb_info, pdb_file);
  if (ec)
    return nullptr;

  // if the file doesn't exist, is not a pdb, or doesn't have a matching guid,
  // fail.
  llvm::file_magic magic;
  ec = llvm::identify_magic(pdb_file, magic);
  if (ec || magic != llvm::file_magic::pdb)
    return nullptr;
  std::unique_ptr<PDBFile> pdb = loadPDBFile(pdb_file, allocator);
  auto expected_info = pdb->getPDBInfoStream();
  if (!expected_info) {
    llvm::consumeError(expected_info.takeError());
    return nullptr;
  }
  llvm::codeview::GUID guid;
  memcpy(&guid, pdb_info->PDB70.Signature, 16);

  if (expected_info->getGuid() != guid)
    return nullptr;
  return std::move(pdb);
}

static bool IsFunctionPrologue(const CompilandIndexItem &cci,
                               lldb::addr_t addr) {
  // FIXME: Implement this.
  return false;
}

static bool IsFunctionEpilogue(const CompilandIndexItem &cci,
                               lldb::addr_t addr) {
  // FIXME: Implement this.
  return false;
}

void SymbolFileNativePDB::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                DebuggerInitialize);
}

void SymbolFileNativePDB::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

void SymbolFileNativePDB::DebuggerInitialize(lldb_private::Debugger &debugger) {
}

lldb_private::ConstString SymbolFileNativePDB::GetPluginNameStatic() {
  static ConstString g_name("native-pdb");
  return g_name;
}

const char *SymbolFileNativePDB::GetPluginDescriptionStatic() {
  return "Microsoft PDB debug symbol cross-platform file reader.";
}

lldb_private::SymbolFile *
SymbolFileNativePDB::CreateInstance(lldb_private::ObjectFile *obj_file) {
  return new SymbolFileNativePDB(obj_file);
}

SymbolFileNativePDB::SymbolFileNativePDB(lldb_private::ObjectFile *object_file)
    : SymbolFile(object_file) {}

SymbolFileNativePDB::~SymbolFileNativePDB() {}

uint32_t SymbolFileNativePDB::CalculateAbilities() {
  uint32_t abilities = 0;
  if (!m_obj_file)
    return 0;

  if (!m_index) {
    // Lazily load and match the PDB file, but only do this once.
    std::unique_ptr<PDBFile> file_up =
        loadMatchingPDBFile(m_obj_file->GetFileSpec().GetPath(), m_allocator);

    if (!file_up) {
      auto module_sp = m_obj_file->GetModule();
      if (!module_sp)
        return 0;
      // See if any symbol file is specified through `--symfile` option.
      FileSpec symfile = module_sp->GetSymbolFileFileSpec();
      if (!symfile)
        return 0;
      file_up = loadPDBFile(symfile.GetPath(), m_allocator);
    }

    if (!file_up)
      return 0;

    auto expected_index = PdbIndex::create(std::move(file_up));
    if (!expected_index) {
      llvm::consumeError(expected_index.takeError());
      return 0;
    }
    m_index = std::move(*expected_index);
  }
  if (!m_index)
    return 0;

  // We don't especially have to be precise here.  We only distinguish between
  // stripped and not stripped.
  abilities = kAllAbilities;

  if (m_index->dbi().isStripped())
    abilities &= ~(Blocks | LocalVariables);
  return abilities;
}

void SymbolFileNativePDB::InitializeObject() {
  m_obj_load_address = m_obj_file->GetFileOffset();
  m_index->SetLoadAddress(m_obj_load_address);
  m_index->ParseSectionContribs();
}

uint32_t SymbolFileNativePDB::GetNumCompileUnits() {
  const DbiModuleList &modules = m_index->dbi().modules();
  uint32_t count = modules.getModuleCount();
  if (count == 0)
    return count;

  // The linker can inject an additional "dummy" compilation unit into the
  // PDB. Ignore this special compile unit for our purposes, if it is there.
  // It is always the last one.
  DbiModuleDescriptor last = modules.getModuleDescriptor(count - 1);
  if (last.getModuleName() == "* Linker *")
    --count;
  return count;
}

lldb::FunctionSP SymbolFileNativePDB::CreateFunction(PdbSymUid func_uid,
                                                     const SymbolContext &sc) {
  lldbassert(func_uid.tag() == PDB_SymType::Function);

  PdbSymUid cuid = PdbSymUid::makeCompilandId(func_uid.asCuSym().modi);

  const CompilandIndexItem *cci = m_index->compilands().GetCompiland(cuid);
  lldbassert(cci);
  CVSymbol sym_record =
      cci->m_debug_stream.readSymbolAtOffset(func_uid.asCuSym().offset);

  lldbassert(sym_record.kind() == S_LPROC32 || sym_record.kind() == S_GPROC32);
  SegmentOffsetLength sol = GetSegmentOffsetAndLength(sym_record);

  auto file_vm_addr = m_index->MakeVirtualAddress(sol.so);
  if (file_vm_addr == LLDB_INVALID_ADDRESS || file_vm_addr == 0)
    return nullptr;

  AddressRange func_range(file_vm_addr, sol.length,
                          sc.module_sp->GetSectionList());
  if (!func_range.GetBaseAddress().IsValid())
    return nullptr;

  lldb_private::Type *func_type = nullptr;

  // FIXME: Resolve types and mangled names.
  PdbSymUid sig_uid =
      PdbSymUid::makeTypeSymId(PDB_SymType::FunctionSig, TypeIndex{0}, false);
  Mangled mangled(getSymbolName(sym_record));

  FunctionSP func_sp = std::make_shared<Function>(
      sc.comp_unit, func_uid.toOpaqueId(), sig_uid.toOpaqueId(), mangled,
      func_type, func_range);

  sc.comp_unit->AddFunction(func_sp);
  return func_sp;
}

CompUnitSP
SymbolFileNativePDB::CreateCompileUnit(const CompilandIndexItem &cci) {
  lldb::LanguageType lang =
      cci.m_compile_opts ? TranslateLanguage(cci.m_compile_opts->getLanguage())
                         : lldb::eLanguageTypeUnknown;

  LazyBool optimized = eLazyBoolNo;
  if (cci.m_compile_opts && cci.m_compile_opts->hasOptimizations())
    optimized = eLazyBoolYes;

  llvm::StringRef source_file_name =
      m_index->compilands().GetMainSourceFile(cci);
  lldb_private::FileSpec fs(source_file_name, false);

  CompUnitSP cu_sp =
      std::make_shared<CompileUnit>(m_obj_file->GetModule(), nullptr, fs,
                                    cci.m_uid.toOpaqueId(), lang, optimized);

  const PdbCompilandId &cuid = cci.m_uid.asCompiland();
  m_obj_file->GetModule()->GetSymbolVendor()->SetCompileUnitAtIndex(cuid.modi,
                                                                    cu_sp);
  return cu_sp;
}

FunctionSP SymbolFileNativePDB::GetOrCreateFunction(PdbSymUid func_uid,
                                                    const SymbolContext &sc) {
  lldbassert(func_uid.tag() == PDB_SymType::Function);
  auto emplace_result = m_functions.try_emplace(func_uid.toOpaqueId(), nullptr);
  if (emplace_result.second)
    emplace_result.first->second = CreateFunction(func_uid, sc);

  lldbassert(emplace_result.first->second);
  return emplace_result.first->second;
}

CompUnitSP
SymbolFileNativePDB::GetOrCreateCompileUnit(const CompilandIndexItem &cci) {
  auto emplace_result =
      m_compilands.try_emplace(cci.m_uid.toOpaqueId(), nullptr);
  if (emplace_result.second)
    emplace_result.first->second = CreateCompileUnit(cci);

  lldbassert(emplace_result.first->second);
  return emplace_result.first->second;
}

lldb::CompUnitSP SymbolFileNativePDB::ParseCompileUnitAtIndex(uint32_t index) {
  if (index >= GetNumCompileUnits())
    return CompUnitSP();
  lldbassert(index < UINT16_MAX);
  if (index >= UINT16_MAX)
    return nullptr;

  CompilandIndexItem &item = m_index->compilands().GetOrCreateCompiland(index);

  return GetOrCreateCompileUnit(item);
}

lldb::LanguageType SymbolFileNativePDB::ParseCompileUnitLanguage(
    const lldb_private::SymbolContext &sc) {
  // What fields should I expect to be filled out on the SymbolContext?  Is it
  // safe to assume that `sc.comp_unit` is valid?
  if (!sc.comp_unit)
    return lldb::eLanguageTypeUnknown;
  PdbSymUid uid = PdbSymUid::fromOpaqueId(sc.comp_unit->GetID());
  lldbassert(uid.tag() == PDB_SymType::Compiland);

  CompilandIndexItem *item = m_index->compilands().GetCompiland(uid);
  lldbassert(item);
  if (!item->m_compile_opts)
    return lldb::eLanguageTypeUnknown;

  return TranslateLanguage(item->m_compile_opts->getLanguage());
}

size_t SymbolFileNativePDB::ParseCompileUnitFunctions(
    const lldb_private::SymbolContext &sc) {
  lldbassert(sc.comp_unit);
  return false;
}

static bool NeedsResolvedCompileUnit(uint32_t resolve_scope) {
  // If any of these flags are set, we need to resolve the compile unit.
  uint32_t flags = eSymbolContextCompUnit;
  flags |= eSymbolContextVariable;
  flags |= eSymbolContextFunction;
  flags |= eSymbolContextBlock;
  flags |= eSymbolContextLineEntry;
  return (resolve_scope & flags) != 0;
}

uint32_t
SymbolFileNativePDB::ResolveSymbolContext(const lldb_private::Address &addr,
                                          uint32_t resolve_scope,
                                          lldb_private::SymbolContext &sc) {
  uint32_t resolved_flags = 0;
  lldb::addr_t file_addr = addr.GetFileAddress();

  if (NeedsResolvedCompileUnit(resolve_scope)) {
    llvm::Optional<uint16_t> modi = m_index->GetModuleIndexForVa(file_addr);
    if (!modi)
      return 0;
    PdbSymUid cuid = PdbSymUid::makeCompilandId(*modi);
    CompilandIndexItem *cci = m_index->compilands().GetCompiland(cuid);
    if (!cci)
      return 0;

    sc.comp_unit = GetOrCreateCompileUnit(*cci).get();
    resolved_flags |= eSymbolContextCompUnit;
  }

  if (resolve_scope & eSymbolContextFunction) {
    lldbassert(sc.comp_unit);
    std::vector<SymbolAndUid> matches = m_index->FindSymbolsByVa(file_addr);
    for (const auto &match : matches) {
      if (match.uid.tag() != PDB_SymType::Function)
        continue;
      sc.function = GetOrCreateFunction(match.uid, sc).get();
    }
    resolved_flags |= eSymbolContextFunction;
  }

  if (resolve_scope & eSymbolContextLineEntry) {
    lldbassert(sc.comp_unit);
    if (auto *line_table = sc.comp_unit->GetLineTable()) {
      if (line_table->FindLineEntryByAddress(addr, sc.line_entry))
        resolved_flags |= eSymbolContextLineEntry;
    }
  }

  return resolved_flags;
}

static void AppendLineEntryToSequence(LineTable &table, LineSequence &sequence,
                                      const CompilandIndexItem &cci,
                                      lldb::addr_t base_addr,
                                      uint32_t file_number,
                                      const LineFragmentHeader &block,
                                      const LineNumberEntry &cur) {
  LineInfo cur_info(cur.Flags);

  if (cur_info.isAlwaysStepInto() || cur_info.isNeverStepInto())
    return;

  uint64_t addr = base_addr + cur.Offset;

  bool is_statement = cur_info.isStatement();
  bool is_prologue = IsFunctionPrologue(cci, addr);
  bool is_epilogue = IsFunctionEpilogue(cci, addr);

  uint32_t lno = cur_info.getStartLine();

  table.AppendLineEntryToSequence(&sequence, addr, lno, 0, file_number,
                                  is_statement, false, is_prologue, is_epilogue,
                                  false);
}

static void TerminateLineSequence(LineTable &table,
                                  const LineFragmentHeader &block,
                                  lldb::addr_t base_addr, uint32_t file_number,
                                  uint32_t last_line,
                                  std::unique_ptr<LineSequence> seq) {
  // The end is always a terminal entry, so insert it regardless.
  table.AppendLineEntryToSequence(seq.get(), base_addr + block.CodeSize,
                                  last_line, 0, file_number, false, false,
                                  false, false, true);
  table.InsertSequence(seq.release());
}

bool SymbolFileNativePDB::ParseCompileUnitLineTable(
    const lldb_private::SymbolContext &sc) {
  // Unfortunately LLDB is set up to parse the entire compile unit line table
  // all at once, even if all it really needs is line info for a specific
  // function.  In the future it would be nice if it could set the sc.m_function
  // member, and we could only get the line info for the function in question.
  lldbassert(sc.comp_unit);
  PdbSymUid cu_id = PdbSymUid::fromOpaqueId(sc.comp_unit->GetID());
  lldbassert(cu_id.isCompiland());
  CompilandIndexItem *cci = m_index->compilands().GetCompiland(cu_id);
  lldbassert(cci);
  auto line_table = llvm::make_unique<LineTable>(sc.comp_unit);

  // This is basically a copy of the .debug$S subsections from all original COFF
  // object files merged together with address relocations applied.  We are
  // looking for all DEBUG_S_LINES subsections.
  for (const DebugSubsectionRecord &dssr :
       cci->m_debug_stream.getSubsectionsArray()) {
    if (dssr.kind() != DebugSubsectionKind::Lines)
      continue;

    DebugLinesSubsectionRef lines;
    llvm::BinaryStreamReader reader(dssr.getRecordData());
    if (auto EC = lines.initialize(reader)) {
      llvm::consumeError(std::move(EC));
      return false;
    }

    const LineFragmentHeader *lfh = lines.header();
    uint64_t virtual_addr =
        m_index->MakeVirtualAddress(lfh->RelocSegment, lfh->RelocOffset);

    const auto &checksums = cci->m_strings.checksums().getArray();
    const auto &strings = cci->m_strings.strings();
    for (const LineColumnEntry &group : lines) {
      // Indices in this structure are actually offsets of records in the
      // DEBUG_S_FILECHECKSUMS subsection.  Those entries then have an index
      // into the global PDB string table.
      auto iter = checksums.at(group.NameIndex);
      if (iter == checksums.end())
        continue;

      llvm::Expected<llvm::StringRef> efn =
          strings.getString(iter->FileNameOffset);
      if (!efn) {
        llvm::consumeError(efn.takeError());
        continue;
      }

      // LLDB wants the index of the file in the list of support files.
      auto fn_iter = llvm::find(cci->m_file_list, *efn);
      lldbassert(fn_iter != cci->m_file_list.end());
      uint32_t file_index = std::distance(cci->m_file_list.begin(), fn_iter);

      std::unique_ptr<LineSequence> sequence(
          line_table->CreateLineSequenceContainer());
      lldbassert(!group.LineNumbers.empty());

      for (const LineNumberEntry &entry : group.LineNumbers) {
        AppendLineEntryToSequence(*line_table, *sequence, *cci, virtual_addr,
                                  file_index, *lfh, entry);
      }
      LineInfo last_line(group.LineNumbers.back().Flags);
      TerminateLineSequence(*line_table, *lfh, virtual_addr, file_index,
                            last_line.getEndLine(), std::move(sequence));
    }
  }

  if (line_table->GetSize() == 0)
    return false;

  sc.comp_unit->SetLineTable(line_table.release());
  return true;
}

bool SymbolFileNativePDB::ParseCompileUnitDebugMacros(
    const lldb_private::SymbolContext &sc) {
  // PDB doesn't contain information about macros
  return false;
}

bool SymbolFileNativePDB::ParseCompileUnitSupportFiles(
    const lldb_private::SymbolContext &sc,
    lldb_private::FileSpecList &support_files) {
  lldbassert(sc.comp_unit);

  PdbSymUid comp_uid = PdbSymUid::fromOpaqueId(sc.comp_unit->GetID());
  lldbassert(comp_uid.tag() == PDB_SymType::Compiland);

  const CompilandIndexItem *cci = m_index->compilands().GetCompiland(comp_uid);
  lldbassert(cci);

  for (llvm::StringRef f : cci->m_file_list) {
    FileSpec spec(f, false, FileSpec::Style::windows);
    support_files.Append(spec);
  }

  return true;
}

bool SymbolFileNativePDB::ParseImportedModules(
    const lldb_private::SymbolContext &sc,
    std::vector<lldb_private::ConstString> &imported_modules) {
  // PDB does not yet support module debug info
  return false;
}

size_t SymbolFileNativePDB::ParseFunctionBlocks(
    const lldb_private::SymbolContext &sc) {
  lldbassert(sc.comp_unit && sc.function);
  return 0;
}

uint32_t SymbolFileNativePDB::FindFunctions(
    const lldb_private::ConstString &name,
    const lldb_private::CompilerDeclContext *parent_decl_ctx,
    uint32_t name_type_mask, bool include_inlines, bool append,
    lldb_private::SymbolContextList &sc_list) {
  // For now we only support lookup by method name.
  if (!(name_type_mask & eFunctionNameTypeMethod))
    return 0;

  using SymbolAndOffset = std::pair<uint32_t, llvm::codeview::CVSymbol>;

  std::vector<SymbolAndOffset> matches = m_index->globals().findRecordsByName(
      name.GetStringRef(), m_index->symrecords());
  for (const SymbolAndOffset &match : matches) {
    if (match.second.kind() != S_PROCREF && match.second.kind() != S_LPROCREF)
      continue;
    ProcRefSym proc(match.second.kind());
    cantFail(SymbolDeserializer::deserializeAs<ProcRefSym>(match.second, proc));

    if (!IsValidRecord(proc))
      continue;

    PdbSymUid cuid = PdbSymUid::makeCompilandId(proc);
    CompilandIndexItem &cci = m_index->compilands().GetOrCreateCompiland(cuid);
    lldb_private::SymbolContext sc;

    sc.comp_unit = GetOrCreateCompileUnit(cci).get();
    sc.module_sp = sc.comp_unit->GetModule();
    PdbSymUid func_uid = PdbSymUid::makeCuSymId(proc);
    sc.function = GetOrCreateFunction(func_uid, sc).get();

    sc_list.Append(sc);
  }

  return sc_list.GetSize();
}

uint32_t
SymbolFileNativePDB::FindFunctions(const lldb_private::RegularExpression &regex,
                                   bool include_inlines, bool append,
                                   lldb_private::SymbolContextList &sc_list) {
  return 0;
}

lldb_private::CompilerDeclContext SymbolFileNativePDB::FindNamespace(
    const lldb_private::SymbolContext &sc,
    const lldb_private::ConstString &name,
    const lldb_private::CompilerDeclContext *parent_decl_ctx) {
  return {};
}

lldb_private::TypeSystem *
SymbolFileNativePDB::GetTypeSystemForLanguage(lldb::LanguageType language) {
  auto type_system =
      m_obj_file->GetModule()->GetTypeSystemForLanguage(language);
  if (type_system)
    type_system->SetSymbolFile(this);
  return type_system;
}

lldb_private::ConstString SymbolFileNativePDB::GetPluginName() {
  static ConstString g_name("pdb");
  return g_name;
}

uint32_t SymbolFileNativePDB::GetPluginVersion() { return 1; }
