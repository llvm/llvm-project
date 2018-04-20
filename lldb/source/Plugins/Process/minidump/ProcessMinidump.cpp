//===-- ProcessMinidump.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "ProcessMinidump.h"
#include "ThreadMinidump.h"

// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/DataBufferLLVM.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Log.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"

// C includes
// C++ includes

using namespace lldb;
using namespace lldb_private;
using namespace minidump;

//------------------------------------------------------------------
/// A placeholder module used for minidumps, where the original
/// object files may not be available (so we can't parse the object
/// files to extract the set of sections/segments)
///
/// This placeholder module has a single synthetic section (.module_image)
/// which represents the module memory range covering the whole module.
//------------------------------------------------------------------
class PlaceholderModule : public Module {
public:
  PlaceholderModule(const FileSpec &file_spec, const ArchSpec &arch) :
    Module(file_spec, arch) {}

  // Creates a synthetic module section covering the whole module image
  // (and sets the section load address as well)
  void CreateImageSection(const MinidumpModule *module, Target& target) {
    const ConstString section_name(".module_image");
    lldb::SectionSP section_sp(new Section(
        shared_from_this(),     // Module to which this section belongs.
        nullptr,                // ObjectFile
        0,                      // Section ID.
        section_name,           // Section name.
        eSectionTypeContainer,  // Section type.
        module->base_of_image,  // VM address.
        module->size_of_image,  // VM size in bytes of this section.
        0,                      // Offset of this section in the file.
        module->size_of_image,  // Size of the section as found in the file.
        12,                     // Alignment of the section (log2)
        0,                      // Flags for this section.
        1));                    // Number of host bytes per target byte
    section_sp->SetPermissions(ePermissionsExecutable | ePermissionsReadable);
    GetSectionList()->AddSection(section_sp);
    target.GetSectionLoadList().SetSectionLoadAddress(
        section_sp, module->base_of_image);
  }

  ObjectFile *GetObjectFile() override { return nullptr; }

  SectionList *GetSectionList() override {
    return Module::GetUnifiedSectionList();
  }
};

ConstString ProcessMinidump::GetPluginNameStatic() {
  static ConstString g_name("minidump");
  return g_name;
}

const char *ProcessMinidump::GetPluginDescriptionStatic() {
  return "Minidump plug-in.";
}

lldb::ProcessSP ProcessMinidump::CreateInstance(lldb::TargetSP target_sp,
                                                lldb::ListenerSP listener_sp,
                                                const FileSpec *crash_file) {
  if (!crash_file)
    return nullptr;

  lldb::ProcessSP process_sp;
  // Read enough data for the Minidump header
  constexpr size_t header_size = sizeof(MinidumpHeader);
  auto DataPtr =
      DataBufferLLVM::CreateSliceFromPath(crash_file->GetPath(), header_size, 0);
  if (!DataPtr)
    return nullptr;

  assert(DataPtr->GetByteSize() == header_size);

  // first, only try to parse the header, beacuse we need to be fast
  llvm::ArrayRef<uint8_t> HeaderBytes = DataPtr->GetData();
  const MinidumpHeader *header = MinidumpHeader::Parse(HeaderBytes);
  if (header == nullptr)
    return nullptr;

  auto AllData = DataBufferLLVM::CreateSliceFromPath(crash_file->GetPath(), -1, 0);
  if (!AllData)
    return nullptr;

  auto minidump_parser = MinidumpParser::Create(AllData);
  // check if the parser object is valid
  if (!minidump_parser)
    return nullptr;

  return std::make_shared<ProcessMinidump>(target_sp, listener_sp, *crash_file,
                                           minidump_parser.getValue());
}

bool ProcessMinidump::CanDebug(lldb::TargetSP target_sp,
                               bool plugin_specified_by_name) {
  return true;
}

ProcessMinidump::ProcessMinidump(lldb::TargetSP target_sp,
                                 lldb::ListenerSP listener_sp,
                                 const FileSpec &core_file,
                                 MinidumpParser minidump_parser)
    : Process(target_sp, listener_sp), m_minidump_parser(minidump_parser),
      m_core_file(core_file), m_is_wow64(false) {}

ProcessMinidump::~ProcessMinidump() {
  Clear();
  // We need to call finalize on the process before destroying ourselves
  // to make sure all of the broadcaster cleanup goes as planned. If we
  // destruct this class, then Process::~Process() might have problems
  // trying to fully destroy the broadcaster.
  Finalize();
}

void ProcessMinidump::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  ProcessMinidump::CreateInstance);
  });
}

void ProcessMinidump::Terminate() {
  PluginManager::UnregisterPlugin(ProcessMinidump::CreateInstance);
}

Status ProcessMinidump::DoLoadCore() {
  Status error;

  m_thread_list = m_minidump_parser.GetThreads();
  m_active_exception = m_minidump_parser.GetExceptionStream();
  ReadModuleList();
  GetTarget().SetArchitecture(GetArchitecture());

  llvm::Optional<lldb::pid_t> pid = m_minidump_parser.GetPid();
  if (!pid) {
    error.SetErrorString("failed to parse PID");
    return error;
  }
  SetID(pid.getValue());

  return error;
}

DynamicLoader *ProcessMinidump::GetDynamicLoader() {
  if (m_dyld_ap.get() == nullptr)
    m_dyld_ap.reset(DynamicLoader::FindPlugin(this, nullptr));
  return m_dyld_ap.get();
}

ConstString ProcessMinidump::GetPluginName() { return GetPluginNameStatic(); }

uint32_t ProcessMinidump::GetPluginVersion() { return 1; }

Status ProcessMinidump::DoDestroy() { return Status(); }

void ProcessMinidump::RefreshStateAfterStop() {
  if (!m_active_exception)
    return;

  if (m_active_exception->exception_record.exception_code ==
      MinidumpException::DumpRequested) {
    return;
  }

  lldb::StopInfoSP stop_info;
  lldb::ThreadSP stop_thread;

  Process::m_thread_list.SetSelectedThreadByID(m_active_exception->thread_id);
  stop_thread = Process::m_thread_list.GetSelectedThread();
  ArchSpec arch = GetArchitecture();

  if (arch.GetTriple().getOS() == llvm::Triple::Linux) {
    stop_info = StopInfo::CreateStopReasonWithSignal(
        *stop_thread, m_active_exception->exception_record.exception_code);
  } else {
    std::string desc;
    llvm::raw_string_ostream desc_stream(desc);
    desc_stream << "Exception "
                << llvm::format_hex(
                       m_active_exception->exception_record.exception_code, 8)
                << " encountered at address "
                << llvm::format_hex(
                       m_active_exception->exception_record.exception_address,
                       8);
    stop_info = StopInfo::CreateStopReasonWithException(
        *stop_thread, desc_stream.str().c_str());
  }

  stop_thread->SetStopInfo(stop_info);
}

bool ProcessMinidump::IsAlive() { return true; }

bool ProcessMinidump::WarnBeforeDetach() const { return false; }

size_t ProcessMinidump::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                   Status &error) {
  // Don't allow the caching that lldb_private::Process::ReadMemory does
  // since we have it all cached in our dump file anyway.
  return DoReadMemory(addr, buf, size, error);
}

size_t ProcessMinidump::DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                     Status &error) {

  llvm::ArrayRef<uint8_t> mem = m_minidump_parser.GetMemory(addr, size);
  if (mem.empty()) {
    error.SetErrorString("could not parse memory info");
    return 0;
  }

  std::memcpy(buf, mem.data(), mem.size());
  return mem.size();
}

ArchSpec ProcessMinidump::GetArchitecture() {
  if (!m_is_wow64) {
    return m_minidump_parser.GetArchitecture();
  }

  llvm::Triple triple;
  triple.setVendor(llvm::Triple::VendorType::UnknownVendor);
  triple.setArch(llvm::Triple::ArchType::x86);
  triple.setOS(llvm::Triple::OSType::Win32);
  return ArchSpec(triple);
}

Status ProcessMinidump::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                            MemoryRegionInfo &range_info) {
  Status error;
  auto info = m_minidump_parser.GetMemoryRegionInfo(load_addr);
  if (!info) {
    error.SetErrorString("No valid MemoryRegionInfo found!");
    return error;
  }
  range_info = info.getValue();
  return error;
}

void ProcessMinidump::Clear() { Process::m_thread_list.Clear(); }

bool ProcessMinidump::UpdateThreadList(ThreadList &old_thread_list,
                                       ThreadList &new_thread_list) {
  uint32_t num_threads = 0;
  if (m_thread_list.size() > 0)
    num_threads = m_thread_list.size();

  for (lldb::tid_t tid = 0; tid < num_threads; ++tid) {
    llvm::ArrayRef<uint8_t> context;
    if (!m_is_wow64)
      context = m_minidump_parser.GetThreadContext(m_thread_list[tid]);
    else
      context = m_minidump_parser.GetThreadContextWow64(m_thread_list[tid]);

    lldb::ThreadSP thread_sp(
        new ThreadMinidump(*this, m_thread_list[tid], context));
    new_thread_list.AddThread(thread_sp);
  }
  return new_thread_list.GetSize(false) > 0;
}

void ProcessMinidump::ReadModuleList() {
  std::vector<const MinidumpModule *> filtered_modules =
      m_minidump_parser.GetFilteredModuleList();

  for (auto module : filtered_modules) {
    llvm::Optional<std::string> name =
        m_minidump_parser.GetMinidumpString(module->module_name_rva);

    if (!name)
      continue;

    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_MODULES));
    if (log) {
      log->Printf("ProcessMinidump::%s found module: name: %s %#010" PRIx64
                  "-%#010" PRIx64 " size: %" PRIu32,
                  __FUNCTION__, name.getValue().c_str(),
                  uint64_t(module->base_of_image),
                  module->base_of_image + module->size_of_image,
                  uint32_t(module->size_of_image));
    }

    // check if the process is wow64 - a 32 bit windows process running on a
    // 64 bit windows
    if (llvm::StringRef(name.getValue()).endswith_lower("wow64.dll")) {
      m_is_wow64 = true;
    }

    const auto file_spec =
        FileSpec(name.getValue(), true, GetArchitecture().GetTriple());
    ModuleSpec module_spec = file_spec;
    Status error;
    lldb::ModuleSP module_sp = GetTarget().GetSharedModule(module_spec, &error);
    if (!module_sp || error.Fail()) {
      // We failed to locate a matching local object file. Fortunately,
      // the minidump format encodes enough information about each module's
      // memory range to allow us to create placeholder modules.
      //
      // This enables most LLDB functionality involving address-to-module
      // translations (ex. identifing the module for a stack frame PC) and
      // modules/sections commands (ex. target modules list, ...)
      auto placeholder_module =
          std::make_shared<PlaceholderModule>(file_spec, GetArchitecture());
      placeholder_module->CreateImageSection(module, GetTarget());
      module_sp = placeholder_module;
      GetTarget().GetImages().Append(module_sp);
    }

    if (log) {
      log->Printf("ProcessMinidump::%s load module: name: %s", __FUNCTION__,
                  name.getValue().c_str());
    }

    bool load_addr_changed = false;
    module_sp->SetLoadAddress(GetTarget(), module->base_of_image, false,
                              load_addr_changed);
  }
}

bool ProcessMinidump::GetProcessInfo(ProcessInstanceInfo &info) {
  info.Clear();
  info.SetProcessID(GetID());
  info.SetArchitecture(GetArchitecture());
  lldb::ModuleSP module_sp = GetTarget().GetExecutableModule();
  if (module_sp) {
    const bool add_exe_file_as_first_arg = false;
    info.SetExecutableFile(GetTarget().GetExecutableModule()->GetFileSpec(),
                           add_exe_file_as_first_arg);
  }
  return true;
}
