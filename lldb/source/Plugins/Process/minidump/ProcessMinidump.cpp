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
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/LLDBAssert.h"

// C includes
// C++ includes

using namespace lldb_private;
using namespace minidump;

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
  const size_t header_size = sizeof(MinidumpHeader);
  lldb::DataBufferSP data_sp(crash_file->MemoryMapFileContents(0, header_size));
  // The memory map can fail
  if (!data_sp)
    return nullptr;

  // first, only try to parse the header, beacuse we need to be fast
  llvm::ArrayRef<uint8_t> header_data(data_sp->GetBytes(), header_size);
  const MinidumpHeader *header = MinidumpHeader::Parse(header_data);

  if (data_sp->GetByteSize() != header_size || header == nullptr)
    return nullptr;

  lldb::DataBufferSP all_data_sp(crash_file->MemoryMapFileContents());
  auto minidump_parser = MinidumpParser::Create(all_data_sp);
  // check if the parser object is valid
  // skip if the Minidump file is Windows generated, because we are still
  // work-in-progress
  if (!minidump_parser ||
      minidump_parser->GetArchitecture().GetTriple().getOS() ==
          llvm::Triple::OSType::Win32)
    return nullptr;

  return lldb::ProcessSP(new ProcessMinidump(
      target_sp, listener_sp, *crash_file, minidump_parser.getValue()));
}

// TODO leave it to be only "return true" ?
bool ProcessMinidump::CanDebug(lldb::TargetSP target_sp,
                               bool plugin_specified_by_name) {
  return true;
}

ProcessMinidump::ProcessMinidump(lldb::TargetSP target_sp,
                                 lldb::ListenerSP listener_sp,
                                 const FileSpec &core_file,
                                 MinidumpParser minidump_parser)
    : Process(target_sp, listener_sp), m_minidump_parser(minidump_parser),
      m_core_file(core_file) {}

ProcessMinidump::~ProcessMinidump() {
  Clear();
  // We need to call finalize on the process before destroying ourselves
  // to make sure all of the broadcaster cleanup goes as planned. If we
  // destruct this class, then Process::~Process() might have problems
  // trying to fully destroy the broadcaster.
  Finalize();
}

void ProcessMinidump::Initialize() {
  static std::once_flag g_once_flag;

  std::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  ProcessMinidump::CreateInstance);
  });
}

void ProcessMinidump::Terminate() {
  PluginManager::UnregisterPlugin(ProcessMinidump::CreateInstance);
}

Error ProcessMinidump::DoLoadCore() {
  Error error;

  m_thread_list = m_minidump_parser.GetThreads();
  m_active_exception = m_minidump_parser.GetExceptionStream();
  GetTarget().SetArchitecture(GetArchitecture());
  ReadModuleList();

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

Error ProcessMinidump::DoDestroy() { return Error(); }

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
                                   lldb_private::Error &error) {
  // Don't allow the caching that lldb_private::Process::ReadMemory does
  // since we have it all cached in our dump file anyway.
  return DoReadMemory(addr, buf, size, error);
}

size_t ProcessMinidump::DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                     lldb_private::Error &error) {
  // I don't have a sense of how frequently this is called or how many memory
  // ranges a Minidump typically has, so I'm not sure if searching for the
  // appropriate range linearly each time is stupid.  Perhaps we should build
  // an index for faster lookups.
  llvm::Optional<Range> range = m_minidump_parser.FindMemoryRange(addr);
  if (!range)
    return 0;

  // There's at least some overlap between the beginning of the desired range
  // (addr) and the current range.  Figure out where the overlap begins and
  // how much overlap there is, then copy it to the destination buffer.
  lldbassert(range->start <= addr);
  const size_t offset = addr - range->start;
  lldbassert(offset < range->range_ref.size());
  const size_t overlap = std::min(size, range->range_ref.size() - offset);
  std::memcpy(buf, range->range_ref.data() + offset, overlap);
  return overlap;
}

ArchSpec ProcessMinidump::GetArchitecture() {
  return m_minidump_parser.GetArchitecture();
}

// TODO - parse the MemoryInfoListStream and implement this method
Error ProcessMinidump::GetMemoryRegionInfo(
    lldb::addr_t load_addr, lldb_private::MemoryRegionInfo &range_info) {
  return {};
}

void ProcessMinidump::Clear() { Process::m_thread_list.Clear(); }

bool ProcessMinidump::UpdateThreadList(
    lldb_private::ThreadList &old_thread_list,
    lldb_private::ThreadList &new_thread_list) {
  uint32_t num_threads = 0;
  if (m_thread_list.size() > 0)
    num_threads = m_thread_list.size();

  for (lldb::tid_t tid = 0; tid < num_threads; ++tid) {
    lldb::ThreadSP thread_sp(new ThreadMinidump(
        *this, m_thread_list[tid],
        m_minidump_parser.GetThreadContext(m_thread_list[tid])));
    new_thread_list.AddThread(thread_sp);
  }
  return new_thread_list.GetSize(false) > 0;
}

void ProcessMinidump::ReadModuleList() {
  llvm::ArrayRef<MinidumpModule> modules = m_minidump_parser.GetModuleList();

  for (auto module : modules) {
    llvm::Optional<std::string> name =
        m_minidump_parser.GetMinidumpString(module.module_name_rva);

    if (!name)
      continue;

    const auto file_spec = FileSpec(name.getValue(), true);
    ModuleSpec module_spec = file_spec;
    Error error;
    lldb::ModuleSP module_sp =
        this->GetTarget().GetSharedModule(module_spec, &error);
    if (!module_sp || error.Fail()) {
      continue;
    }

    bool load_addr_changed = false;
    module_sp->SetLoadAddress(this->GetTarget(), module.base_of_image, false,
                              load_addr_changed);
  }
}
