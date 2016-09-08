//===-- ProcessWinMiniDump.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ProcessWinMiniDump.h"

#include "lldb/Host/windows/windows.h"
#include <DbgHelp.h>

#include <assert.h>
#include <memory>
#include <mutex>
#include <stdlib.h>

#include "Plugins/DynamicLoader/Windows-DYLD/DynamicLoaderWindowsDYLD.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/LLDBAssert.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "Plugins/Process/Windows/Common/NtStructures.h"
#include "Plugins/Process/Windows/Common/ProcessWindowsLog.h"

#include "ExceptionRecord.h"
#include "ThreadWinMiniDump.h"

using namespace lldb_private;

// Implementation class for ProcessWinMiniDump encapsulates the Windows-specific
// code, keeping non-portable types out of the header files.
// TODO(amccarth):  Determine if we need a mutex for access.  Given that this is
// postmortem debugging, I don't think so.
class ProcessWinMiniDump::Impl {
public:
  Impl(const FileSpec &core_file, ProcessWinMiniDump *self);
  ~Impl();

  Error DoLoadCore();

  bool UpdateThreadList(ThreadList &old_thread_list,
                        ThreadList &new_thread_list);

  void RefreshStateAfterStop();

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size, Error &error);

  Error GetMemoryRegionInfo(lldb::addr_t load_addr,
                            lldb_private::MemoryRegionInfo &info);

private:
  // Describes a range of memory captured in the mini dump.
  struct Range {
    lldb::addr_t start; // virtual address of the beginning of the range
    size_t size;        // size of the range in bytes
    const uint8_t *ptr; // absolute pointer to the first byte of the range
  };

  // If the mini dump has a memory range that contains the desired address, it
  // returns true with the details of the range in *range_out.  Otherwise, it
  // returns false.
  bool FindMemoryRange(lldb::addr_t addr, Range *range_out) const;

  lldb_private::Error MapMiniDumpIntoMemory();

  lldb_private::ArchSpec DetermineArchitecture();

  void ReadExceptionRecord();

  void ReadMiscInfo();

  void ReadModuleList();

  // A thin wrapper around WinAPI's MiniDumpReadDumpStream to avoid redundant
  // checks.  If there's a failure (e.g., if the requested stream doesn't
  // exist),
  // the function returns nullptr and sets *size_out to 0.
  void *FindDumpStream(unsigned stream_number, size_t *size_out) const;

  // Getting a string out of a mini dump is a chore.  You're usually given a
  // relative virtual address (RVA), which points to a counted string that's in
  // Windows Unicode (UTF-16).  This wrapper handles all the redirection and
  // returns a UTF-8 copy of the string.
  std::string GetMiniDumpString(RVA rva) const;

  ProcessWinMiniDump *m_self; // non-owning back pointer
  FileSpec m_core_file;
  HANDLE m_dump_file; // handle to the open minidump file
  HANDLE m_mapping;   // handle to the file mapping for the minidump file
  void *m_base_addr;  // base memory address of the minidump
  std::shared_ptr<ExceptionRecord> m_exception_sp;
  bool m_is_wow64; // minidump is of a 32-bit process captured with a 64-bit
                   // debugger
};

ProcessWinMiniDump::Impl::Impl(const FileSpec &core_file,
                               ProcessWinMiniDump *self)
    : m_self(self), m_core_file(core_file), m_dump_file(INVALID_HANDLE_VALUE),
      m_mapping(NULL), m_base_addr(nullptr), m_exception_sp(),
      m_is_wow64(false) {}

ProcessWinMiniDump::Impl::~Impl() {
  if (m_base_addr) {
    ::UnmapViewOfFile(m_base_addr);
    m_base_addr = nullptr;
  }
  if (m_mapping) {
    ::CloseHandle(m_mapping);
    m_mapping = NULL;
  }
  if (m_dump_file != INVALID_HANDLE_VALUE) {
    ::CloseHandle(m_dump_file);
    m_dump_file = INVALID_HANDLE_VALUE;
  }
}

Error ProcessWinMiniDump::Impl::DoLoadCore() {
  Error error = MapMiniDumpIntoMemory();
  if (error.Fail()) {
    return error;
  }

  m_self->GetTarget().SetArchitecture(DetermineArchitecture());
  ReadMiscInfo(); // notably for process ID
  ReadModuleList();
  ReadExceptionRecord();

  return error;
}

bool ProcessWinMiniDump::Impl::UpdateThreadList(ThreadList &old_thread_list,
                                                ThreadList &new_thread_list) {
  size_t size = 0;
  auto thread_list_ptr = static_cast<const MINIDUMP_THREAD_LIST *>(
      FindDumpStream(ThreadListStream, &size));
  if (thread_list_ptr) {
    const ULONG32 thread_count = thread_list_ptr->NumberOfThreads;
    for (ULONG32 i = 0; i < thread_count; ++i) {
      const auto &mini_dump_thread = thread_list_ptr->Threads[i];
      auto thread_sp = std::make_shared<ThreadWinMiniDump>(
          *m_self, mini_dump_thread.ThreadId);
      if (mini_dump_thread.ThreadContext.DataSize >= sizeof(CONTEXT)) {
        const CONTEXT *context = reinterpret_cast<const CONTEXT *>(
            static_cast<const char *>(m_base_addr) +
            mini_dump_thread.ThreadContext.Rva);

        if (m_is_wow64) {
          // On Windows, a 32-bit process can run on a 64-bit machine under
          // WOW64.
          // If the minidump was captured with a 64-bit debugger, then the
          // CONTEXT
          // we just grabbed from the mini_dump_thread is the one for the 64-bit
          // "native" process rather than the 32-bit "guest" process we care
          // about.
          // In this case, we can get the 32-bit CONTEXT from the TEB (Thread
          // Environment Block) of the 64-bit process.
          Error error;
          TEB64 wow64teb = {0};
          m_self->ReadMemory(mini_dump_thread.Teb, &wow64teb, sizeof(wow64teb),
                             error);
          if (error.Success()) {
            // Slot 1 of the thread-local storage in the 64-bit TEB points to a
            // structure
            // that includes the 32-bit CONTEXT (after a ULONG).
            // See:  https://msdn.microsoft.com/en-us/library/ms681670.aspx
            const size_t addr = wow64teb.TlsSlots[1];
            Range range = {0};
            if (FindMemoryRange(addr, &range)) {
              lldbassert(range.start <= addr);
              const size_t offset = addr - range.start + sizeof(ULONG);
              if (offset < range.size) {
                const size_t overlap = range.size - offset;
                if (overlap >= sizeof(CONTEXT)) {
                  context =
                      reinterpret_cast<const CONTEXT *>(range.ptr + offset);
                }
              }
            }
          }

          // NOTE:  We don't currently use the TEB for anything else.  If we
          // need it in
          // the future, the 32-bit TEB is located according to the address
          // stored in the
          // first slot of the 64-bit TEB (wow64teb.Reserved1[0]).
        }

        thread_sp->SetContext(context);
      }
      new_thread_list.AddThread(thread_sp);
    }
  }

  return new_thread_list.GetSize(false) > 0;
}

void ProcessWinMiniDump::Impl::RefreshStateAfterStop() {
  if (!m_exception_sp)
    return;

  auto active_exception = m_exception_sp;
  std::string desc;
  llvm::raw_string_ostream desc_stream(desc);
  desc_stream << "Exception "
              << llvm::format_hex(active_exception->GetExceptionCode(), 8)
              << " encountered at address "
              << llvm::format_hex(active_exception->GetExceptionAddress(), 8);
  m_self->m_thread_list.SetSelectedThreadByID(active_exception->GetThreadID());
  auto stop_thread = m_self->m_thread_list.GetSelectedThread();
  auto stop_info = StopInfo::CreateStopReasonWithException(
      *stop_thread, desc_stream.str().c_str());
  stop_thread->SetStopInfo(stop_info);
}

size_t ProcessWinMiniDump::Impl::DoReadMemory(lldb::addr_t addr, void *buf,
                                              size_t size, Error &error) {
  // I don't have a sense of how frequently this is called or how many memory
  // ranges a mini dump typically has, so I'm not sure if searching for the
  // appropriate range linearly each time is stupid.  Perhaps we should build
  // an index for faster lookups.
  Range range = {0};
  if (!FindMemoryRange(addr, &range)) {
    return 0;
  }

  // There's at least some overlap between the beginning of the desired range
  // (addr) and the current range.  Figure out where the overlap begins and
  // how much overlap there is, then copy it to the destination buffer.
  lldbassert(range.start <= addr);
  const size_t offset = addr - range.start;
  lldbassert(offset < range.size);
  const size_t overlap = std::min(size, range.size - offset);
  std::memcpy(buf, range.ptr + offset, overlap);
  return overlap;
}

Error ProcessWinMiniDump::Impl::GetMemoryRegionInfo(
    lldb::addr_t load_addr, lldb_private::MemoryRegionInfo &info) {
  Error error;
  size_t size;
  const auto list = reinterpret_cast<const MINIDUMP_MEMORY_INFO_LIST *>(
      FindDumpStream(MemoryInfoListStream, &size));
  if (list == nullptr || size < sizeof(MINIDUMP_MEMORY_INFO_LIST)) {
    error.SetErrorString("the mini dump contains no memory range information");
    return error;
  }

  if (list->SizeOfEntry < sizeof(MINIDUMP_MEMORY_INFO)) {
    error.SetErrorString("the entries in the mini dump memory info list are "
                         "smaller than expected");
    return error;
  }

  if (size < list->SizeOfHeader + list->SizeOfEntry * list->NumberOfEntries) {
    error.SetErrorString("the mini dump memory info list is incomplete");
    return error;
  }

  for (int i = 0; i < list->NumberOfEntries; ++i) {
    const auto entry = reinterpret_cast<const MINIDUMP_MEMORY_INFO *>(
        reinterpret_cast<const char *>(list) + list->SizeOfHeader +
        i * list->SizeOfEntry);
    const auto head = entry->BaseAddress;
    const auto tail = head + entry->RegionSize;
    if (head <= load_addr && load_addr < tail) {
      info.SetReadable(IsPageReadable(entry->Protect) ? MemoryRegionInfo::eYes
                                                      : MemoryRegionInfo::eNo);
      info.SetWritable(IsPageWritable(entry->Protect) ? MemoryRegionInfo::eYes
                                                      : MemoryRegionInfo::eNo);
      info.SetExecutable(IsPageExecutable(entry->Protect)
                             ? MemoryRegionInfo::eYes
                             : MemoryRegionInfo::eNo);
      return error;
    }
  }
  // Note that the memory info list doesn't seem to contain ranges in kernel
  // space,
  // so if you're walking a stack that has kernel frames, the stack may appear
  // truncated.
  error.SetErrorString("address is not in a known range");
  return error;
}

bool ProcessWinMiniDump::Impl::FindMemoryRange(lldb::addr_t addr,
                                               Range *range_out) const {
  size_t stream_size = 0;
  auto mem_list_stream = static_cast<const MINIDUMP_MEMORY_LIST *>(
      FindDumpStream(MemoryListStream, &stream_size));
  if (mem_list_stream) {
    for (ULONG32 i = 0; i < mem_list_stream->NumberOfMemoryRanges; ++i) {
      const MINIDUMP_MEMORY_DESCRIPTOR &mem_desc =
          mem_list_stream->MemoryRanges[i];
      const MINIDUMP_LOCATION_DESCRIPTOR &loc_desc = mem_desc.Memory;
      const lldb::addr_t range_start = mem_desc.StartOfMemoryRange;
      const size_t range_size = loc_desc.DataSize;
      if (range_start <= addr && addr < range_start + range_size) {
        range_out->start = range_start;
        range_out->size = range_size;
        range_out->ptr =
            reinterpret_cast<const uint8_t *>(m_base_addr) + loc_desc.Rva;
        return true;
      }
    }
  }

  // Some mini dumps have a Memory64ListStream that captures all the heap
  // memory.  We can't exactly use the same loop as above, because the mini
  // dump uses slightly different data structures to describe those.
  auto mem_list64_stream = static_cast<const MINIDUMP_MEMORY64_LIST *>(
      FindDumpStream(Memory64ListStream, &stream_size));
  if (mem_list64_stream) {
    size_t base_rva = mem_list64_stream->BaseRva;
    for (ULONG32 i = 0; i < mem_list64_stream->NumberOfMemoryRanges; ++i) {
      const MINIDUMP_MEMORY_DESCRIPTOR64 &mem_desc =
          mem_list64_stream->MemoryRanges[i];
      const lldb::addr_t range_start = mem_desc.StartOfMemoryRange;
      const size_t range_size = mem_desc.DataSize;
      if (range_start <= addr && addr < range_start + range_size) {
        range_out->start = range_start;
        range_out->size = range_size;
        range_out->ptr =
            reinterpret_cast<const uint8_t *>(m_base_addr) + base_rva;
        return true;
      }
      base_rva += range_size;
    }
  }

  return false;
}

Error ProcessWinMiniDump::Impl::MapMiniDumpIntoMemory() {
  Error error;
  const char *file = m_core_file.GetCString();
  std::wstring wfile;
  if (!llvm::ConvertUTF8toWide(file, wfile)) {
    error.SetErrorString("Error converting path to UTF-16");
    return error;
  }
  m_dump_file = ::CreateFileW(wfile.c_str(), GENERIC_READ, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (m_dump_file == INVALID_HANDLE_VALUE) {
    error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
  }

  m_mapping =
      ::CreateFileMappingW(m_dump_file, NULL, PAGE_READONLY, 0, 0, NULL);
  if (m_mapping == NULL) {
    error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
  }

  m_base_addr = ::MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0);
  if (m_base_addr == nullptr) {
    error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    return error;
  }

  return error;
}

ArchSpec ProcessWinMiniDump::Impl::DetermineArchitecture() {
  size_t size = 0;
  auto system_info_ptr = static_cast<const MINIDUMP_SYSTEM_INFO *>(
      FindDumpStream(SystemInfoStream, &size));
  if (system_info_ptr) {
    switch (system_info_ptr->ProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_INTEL:
      if (system_info_ptr->ProcessorLevel == 6) {
        return ArchSpec("i686-pc-windows");
      } else {
        return ArchSpec("i386-pc-windows");
      }
      break;
    case PROCESSOR_ARCHITECTURE_AMD64:
      return ArchSpec("x86_64-pc-windows");
    default:
      break;
    }
  }

  return ArchSpec(); // invalid or unknown
}

void ProcessWinMiniDump::Impl::ReadExceptionRecord() {
  size_t size = 0;
  auto exception_stream_ptr = static_cast<MINIDUMP_EXCEPTION_STREAM *>(
      FindDumpStream(ExceptionStream, &size));
  if (exception_stream_ptr) {
    m_exception_sp.reset(new ExceptionRecord(
        exception_stream_ptr->ExceptionRecord, exception_stream_ptr->ThreadId));
  } else {
    WINLOG_IFALL(WINDOWS_LOG_PROCESS, "Minidump has no exception record.");
    // TODO:  See if we can recover the exception from the TEB.
  }
}

void ProcessWinMiniDump::Impl::ReadMiscInfo() {
  size_t size = 0;
  const auto misc_info_ptr =
      static_cast<MINIDUMP_MISC_INFO *>(FindDumpStream(MiscInfoStream, &size));
  if (!misc_info_ptr || size < sizeof(MINIDUMP_MISC_INFO)) {
    return;
  }

  if ((misc_info_ptr->Flags1 & MINIDUMP_MISC1_PROCESS_ID) != 0) {
    // This misc info record has the process ID.
    m_self->SetID(misc_info_ptr->ProcessId);
  }
}

void ProcessWinMiniDump::Impl::ReadModuleList() {
  size_t size = 0;
  auto module_list_ptr = static_cast<MINIDUMP_MODULE_LIST *>(
      FindDumpStream(ModuleListStream, &size));
  if (!module_list_ptr || module_list_ptr->NumberOfModules == 0) {
    return;
  }

  for (ULONG32 i = 0; i < module_list_ptr->NumberOfModules; ++i) {
    const auto &module = module_list_ptr->Modules[i];
    const auto file_name = GetMiniDumpString(module.ModuleNameRva);
    const auto file_spec = FileSpec(file_name, true);
    if (FileSpec::Compare(file_spec, FileSpec("wow64.dll", false), false) ==
        0) {
      WINLOG_IFALL(WINDOWS_LOG_PROCESS, "Minidump is for a WOW64 process.");
      m_is_wow64 = true;
    }
    ModuleSpec module_spec = file_spec;

    lldb::ModuleSP module_sp = m_self->GetTarget().GetSharedModule(module_spec);
    if (!module_sp) {
      continue;
    }
    bool load_addr_changed = false;
    module_sp->SetLoadAddress(m_self->GetTarget(), module.BaseOfImage, false,
                              load_addr_changed);
  }
}

void *ProcessWinMiniDump::Impl::FindDumpStream(unsigned stream_number,
                                               size_t *size_out) const {
  void *stream = nullptr;
  *size_out = 0;

  MINIDUMP_DIRECTORY *dir = nullptr;
  if (::MiniDumpReadDumpStream(m_base_addr, stream_number, &dir, nullptr,
                               nullptr) &&
      dir != nullptr && dir->Location.DataSize > 0) {
    assert(dir->StreamType == stream_number);
    *size_out = dir->Location.DataSize;
    stream = static_cast<void *>(static_cast<char *>(m_base_addr) +
                                 dir->Location.Rva);
  }

  return stream;
}

std::string ProcessWinMiniDump::Impl::GetMiniDumpString(RVA rva) const {
  std::string result;
  if (!m_base_addr) {
    return result;
  }
  auto md_string = reinterpret_cast<const MINIDUMP_STRING *>(
      static_cast<const char *>(m_base_addr) + rva);
  auto source_start = reinterpret_cast<const UTF16 *>(md_string->Buffer);
  const auto source_length = ::wcslen(md_string->Buffer);
  const auto source_end = source_start + source_length;
  result.resize(UNI_MAX_UTF8_BYTES_PER_CODE_POINT *
                source_length); // worst case length
  auto result_start = reinterpret_cast<UTF8 *>(&result[0]);
  const auto result_end = result_start + result.size();
  ConvertUTF16toUTF8(&source_start, source_end, &result_start, result_end,
                     strictConversion);
  const auto result_size =
      std::distance(reinterpret_cast<UTF8 *>(&result[0]), result_start);
  result.resize(result_size); // shrink to actual length
  return result;
}

ConstString ProcessWinMiniDump::GetPluginNameStatic() {
  static ConstString g_name("win-minidump");
  return g_name;
}

const char *ProcessWinMiniDump::GetPluginDescriptionStatic() {
  return "Windows minidump plug-in.";
}

void ProcessWinMiniDump::Terminate() {
  PluginManager::UnregisterPlugin(ProcessWinMiniDump::CreateInstance);
}

lldb::ProcessSP ProcessWinMiniDump::CreateInstance(lldb::TargetSP target_sp,
                                                   lldb::ListenerSP listener_sp,
                                                   const FileSpec *crash_file) {
  lldb::ProcessSP process_sp;
  if (crash_file) {
    process_sp.reset(
        new ProcessWinMiniDump(target_sp, listener_sp, *crash_file));
  }
  return process_sp;
}

bool ProcessWinMiniDump::CanDebug(lldb::TargetSP target_sp,
                                  bool plugin_specified_by_name) {
  // TODO(amccarth):  Eventually, this needs some actual logic.
  return true;
}

ProcessWinMiniDump::ProcessWinMiniDump(lldb::TargetSP target_sp,
                                       lldb::ListenerSP listener_sp,
                                       const FileSpec &core_file)
    : ProcessWindows(target_sp, listener_sp),
      m_impl_up(new Impl(core_file, this)) {}

ProcessWinMiniDump::~ProcessWinMiniDump() {
  Clear();
  // We need to call finalize on the process before destroying ourselves
  // to make sure all of the broadcaster cleanup goes as planned. If we
  // destruct this class, then Process::~Process() might have problems
  // trying to fully destroy the broadcaster.
  Finalize();
}

ConstString ProcessWinMiniDump::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ProcessWinMiniDump::GetPluginVersion() { return 1; }

Error ProcessWinMiniDump::DoLoadCore() { return m_impl_up->DoLoadCore(); }

DynamicLoader *ProcessWinMiniDump::GetDynamicLoader() {
  if (m_dyld_ap.get() == NULL)
    m_dyld_ap.reset(DynamicLoader::FindPlugin(
        this, DynamicLoaderWindowsDYLD::GetPluginNameStatic().GetCString()));
  return m_dyld_ap.get();
}

bool ProcessWinMiniDump::UpdateThreadList(ThreadList &old_thread_list,
                                          ThreadList &new_thread_list) {
  return m_impl_up->UpdateThreadList(old_thread_list, new_thread_list);
}

void ProcessWinMiniDump::RefreshStateAfterStop() {
  if (!m_impl_up)
    return;
  return m_impl_up->RefreshStateAfterStop();
}

Error ProcessWinMiniDump::DoDestroy() { return Error(); }

bool ProcessWinMiniDump::IsAlive() { return true; }

bool ProcessWinMiniDump::WarnBeforeDetach() const {
  // Since this is post-mortem debugging, there's no need to warn the user
  // that quitting the debugger will terminate the process.
  return false;
}

size_t ProcessWinMiniDump::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                      Error &error) {
  // Don't allow the caching that lldb_private::Process::ReadMemory does
  // since we have it all cached our our dump file anyway.
  return DoReadMemory(addr, buf, size, error);
}

size_t ProcessWinMiniDump::DoReadMemory(lldb::addr_t addr, void *buf,
                                        size_t size, Error &error) {
  return m_impl_up->DoReadMemory(addr, buf, size, error);
}

Error ProcessWinMiniDump::GetMemoryRegionInfo(
    lldb::addr_t load_addr, lldb_private::MemoryRegionInfo &info) {
  return m_impl_up->GetMemoryRegionInfo(load_addr, info);
}

void ProcessWinMiniDump::Clear() { m_thread_list.Clear(); }

void ProcessWinMiniDump::Initialize() {
  static std::once_flag g_once_flag;

  std::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

ArchSpec ProcessWinMiniDump::GetArchitecture() {
  // TODO
  return ArchSpec();
}
