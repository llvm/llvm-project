//===-- ProcessFreeBSDKernel.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "Plugins/DynamicLoader/FreeBSD-Kernel/DynamicLoaderFreeBSDKernel.h"
#include "ProcessFreeBSDKernel.h"
#include "ThreadFreeBSDKernel.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ProcessFreeBSDKernel)

ProcessFreeBSDKernel::ProcessFreeBSDKernel(lldb::TargetSP target_sp,
                                           ListenerSP listener_sp, kvm_t *kvm,
                                           const FileSpec &core_file)
    : PostMortemProcess(target_sp, listener_sp, core_file), m_kvm(kvm) {}

ProcessFreeBSDKernel::~ProcessFreeBSDKernel() {
  if (m_kvm)
    kvm_close(m_kvm);
}

lldb::ProcessSP ProcessFreeBSDKernel::CreateInstance(lldb::TargetSP target_sp,
                                                     ListenerSP listener_sp,
                                                     const FileSpec *crash_file,
                                                     bool can_connect) {
  ModuleSP executable = target_sp->GetExecutableModule();
  if (crash_file && !can_connect && executable) {
    kvm_t *kvm =
        kvm_open2(executable->GetFileSpec().GetPath().c_str(),
                  crash_file->GetPath().c_str(), O_RDONLY, nullptr, nullptr);
    if (kvm)
      return std::make_shared<ProcessFreeBSDKernel>(target_sp, listener_sp, kvm,
                                                    *crash_file);
  }
  return nullptr;
}

void ProcessFreeBSDKernel::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

void ProcessFreeBSDKernel::Terminate() {
  PluginManager::UnregisterPlugin(ProcessFreeBSDKernel::CreateInstance);
}

Status ProcessFreeBSDKernel::DoDestroy() { return Status(); }

bool ProcessFreeBSDKernel::CanDebug(lldb::TargetSP target_sp,
                                    bool plugin_specified_by_name) {
  return true;
}

void ProcessFreeBSDKernel::RefreshStateAfterStop() {
  if (!m_printed_unread_message) {
    PrintUnreadMessage();
    m_printed_unread_message = true;
  }
}

bool ProcessFreeBSDKernel::DoUpdateThreadList(ThreadList &old_thread_list,
                                              ThreadList &new_thread_list) {
  if (old_thread_list.GetSize(false) == 0) {
    // Make up the thread the first time this is called so we can set our one
    // and only core thread state up.

    // We cannot construct a thread without a register context as that crashes
    // LLDB but we can construct a process without threads to provide minimal
    // memory reading support.
    switch (GetTarget().GetArchitecture().GetMachine()) {
    case llvm::Triple::aarch64:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      break;
    default:
      return false;
    }

    Status error;

    // struct field offsets are written as symbols so that we don't have
    // to figure them out ourselves
    int32_t offset_p_list = ReadSignedIntegerFromMemory(
        FindSymbol("proc_off_p_list"), 4, -1, error);
    int32_t offset_p_pid =
        ReadSignedIntegerFromMemory(FindSymbol("proc_off_p_pid"), 4, -1, error);
    int32_t offset_p_threads = ReadSignedIntegerFromMemory(
        FindSymbol("proc_off_p_threads"), 4, -1, error);
    int32_t offset_p_comm = ReadSignedIntegerFromMemory(
        FindSymbol("proc_off_p_comm"), 4, -1, error);

    int32_t offset_td_tid = ReadSignedIntegerFromMemory(
        FindSymbol("thread_off_td_tid"), 4, -1, error);
    int32_t offset_td_plist = ReadSignedIntegerFromMemory(
        FindSymbol("thread_off_td_plist"), 4, -1, error);
    int32_t offset_td_pcb = ReadSignedIntegerFromMemory(
        FindSymbol("thread_off_td_pcb"), 4, -1, error);
    int32_t offset_td_oncpu = ReadSignedIntegerFromMemory(
        FindSymbol("thread_off_td_oncpu"), 4, -1, error);
    int32_t offset_td_name = ReadSignedIntegerFromMemory(
        FindSymbol("thread_off_td_name"), 4, -1, error);

    // Fail if we were not able to read any of the offsets.
    if (offset_p_list == -1 || offset_p_pid == -1 || offset_p_threads == -1 ||
        offset_p_comm == -1 || offset_td_tid == -1 || offset_td_plist == -1 ||
        offset_td_pcb == -1 || offset_td_oncpu == -1 || offset_td_name == -1)
      return false;

    // dumptid contains the thread-id of the crashing thread
    // dumppcb contains its PCB
    int32_t dumptid =
        ReadSignedIntegerFromMemory(FindSymbol("dumptid"), 4, -1, error);
    lldb::addr_t dumppcb = FindSymbol("dumppcb");

    // stoppcbs is an array of PCBs on all CPUs.
    // Each element is of size pcb_size.
    int32_t pcbsize =
        ReadSignedIntegerFromMemory(FindSymbol("pcb_size"), 4, -1, error);
    lldb::addr_t stoppcbs = FindSymbol("stoppcbs");

    // from FreeBSD sys/param.h
    constexpr size_t fbsd_maxcomlen = 19;

    // Iterate through a linked list of all processes. New processes are added
    // to the head of this list. Which means that earlier PIDs are actually at
    // the end of the list, so we have to walk it backwards. First collect all
    // the processes in the list order.
    std::vector<lldb::addr_t> process_addrs;
    for (lldb::addr_t proc =
             ReadPointerFromMemory(FindSymbol("allproc"), error);
         proc != 0 && proc != LLDB_INVALID_ADDRESS;
         proc = ReadPointerFromMemory(proc + offset_p_list, error)) {
      process_addrs.push_back(proc);
    }

    // Processes are in the linked list in descending PID order, so we must walk
    // them in reverse to get ascending PID order.
    for (auto proc_it = process_addrs.rbegin(); proc_it != process_addrs.rend();
         ++proc_it) {
      lldb::addr_t proc = *proc_it;
      int32_t pid =
          ReadSignedIntegerFromMemory(proc + offset_p_pid, 4, -1, error);
      // process' command-line string
      char comm[fbsd_maxcomlen + 1];
      ReadCStringFromMemory(proc + offset_p_comm, comm, sizeof(comm), error);

      // Iterate through a linked list of all process' threads
      // the initial thread is found in process' p_threads, subsequent
      // elements are linked via td_plist field
      for (lldb::addr_t td =
               ReadPointerFromMemory(proc + offset_p_threads, error);
           td != 0; td = ReadPointerFromMemory(td + offset_td_plist, error)) {
        int32_t tid =
            ReadSignedIntegerFromMemory(td + offset_td_tid, 4, -1, error);
        lldb::addr_t pcb_addr =
            ReadPointerFromMemory(td + offset_td_pcb, error);
        // whether process was on CPU (-1 if not, otherwise CPU number)
        int32_t oncpu =
            ReadSignedIntegerFromMemory(td + offset_td_oncpu, 4, -2, error);
        // thread name
        char thread_name[fbsd_maxcomlen + 1];
        ReadCStringFromMemory(td + offset_td_name, thread_name,
                              sizeof(thread_name), error);

        // If we failed to read TID, ignore this thread.
        if (tid == -1)
          continue;

        std::string thread_desc = llvm::formatv("(pid {0}) {1}", pid, comm);
        if (*thread_name && strcmp(thread_name, comm)) {
          thread_desc += '/';
          thread_desc += thread_name;
        }

        // Roughly:
        // 1. if the thread crashed, its PCB is going to be at "dumppcb"
        // 2. if the thread was on CPU, its PCB is going to be on the CPU
        // 3. otherwise, its PCB is in the thread struct
        if (tid == dumptid) {
          // NB: dumppcb can be LLDB_INVALID_ADDRESS if reading it failed
          pcb_addr = dumppcb;
          thread_desc += " (crashed)";
        } else if (oncpu != -1) {
          // If we managed to read stoppcbs and pcb_size, use them to find
          // the correct PCB.
          if (stoppcbs != LLDB_INVALID_ADDRESS && pcbsize > 0)
            pcb_addr = stoppcbs + oncpu * pcbsize;
          else
            pcb_addr = LLDB_INVALID_ADDRESS;
          thread_desc += llvm::formatv(" (on CPU {0})", oncpu);
        }

        auto thread =
            new ThreadFreeBSDKernel(*this, tid, pcb_addr, thread_desc);

        if (tid == dumptid)
          thread->SetIsCrashedThread(true);

        new_thread_list.AddThread(static_cast<ThreadSP>(thread));
      }
    }
  } else {
    const uint32_t num_threads = old_thread_list.GetSize(false);
    for (uint32_t i = 0; i < num_threads; ++i)
      new_thread_list.AddThread(old_thread_list.GetThreadAtIndex(i, false));
  }
  return new_thread_list.GetSize(false) > 0;
}

Status ProcessFreeBSDKernel::DoLoadCore() {
  // The core is already loaded by CreateInstance().
  return Status();
}

DynamicLoader *ProcessFreeBSDKernel::GetDynamicLoader() {
  if (m_dyld_up.get() == nullptr)
    m_dyld_up.reset(DynamicLoader::FindPlugin(
        this, DynamicLoaderFreeBSDKernel::GetPluginNameStatic()));
  return m_dyld_up.get();
}

lldb::addr_t ProcessFreeBSDKernel::FindSymbol(const char *name) {
  ModuleSP mod_sp = GetTarget().GetExecutableModule();
  const Symbol *sym = mod_sp->FindFirstSymbolWithNameAndType(ConstString(name));
  return sym ? sym->GetLoadAddress(&GetTarget()) : LLDB_INVALID_ADDRESS;
}

void ProcessFreeBSDKernel::PrintUnreadMessage() {
  Target &target = GetTarget();
  Debugger &debugger = target.GetDebugger();

  if (!debugger.GetCommandInterpreter().IsInteractive())
    return;

  Status error;

  // Find msgbufp symbol (pointer to message buffer)
  lldb::addr_t msgbufp_addr = FindSymbol("msgbufp");
  if (msgbufp_addr == LLDB_INVALID_ADDRESS)
    return;

  // Read the pointer value
  lldb::addr_t msgbufp = ReadPointerFromMemory(msgbufp_addr, error);
  if (!error.Success() || msgbufp == LLDB_INVALID_ADDRESS)
    return;

  // Get the type information for struct msgbuf from DWARF
  TypeQuery query("msgbuf");
  TypeResults results;
  target.GetImages().FindTypes(nullptr, query, results);

  uint64_t offset_msg_ptr = 0;
  uint64_t offset_msg_size = 0;
  uint64_t offset_msg_wseq = 0;
  uint64_t offset_msg_rseq = 0;

  if (results.GetTypeMap().GetSize() > 0) {
    // Found type info - use it to get field offsets
    CompilerType msgbuf_type =
        results.GetTypeMap().GetTypeAtIndex(0)->GetForwardCompilerType();

    uint32_t num_fields = msgbuf_type.GetNumFields();
    int field_found = 0;
    for (uint32_t i = 0; i < num_fields; i++) {
      std::string field_name;
      uint64_t field_offset = 0;

      msgbuf_type.GetFieldAtIndex(i, field_name, &field_offset, nullptr,
                                  nullptr);

      if (field_name == "msg_ptr") {
        offset_msg_ptr = field_offset / 8; // Convert bits to bytes
        field_found++;
      } else if (field_name == "msg_size") {
        offset_msg_size = field_offset / 8;
        field_found++;
      } else if (field_name == "msg_wseq") {
        offset_msg_wseq = field_offset / 8;
        field_found++;
      } else if (field_name == "msg_rseq") {
        offset_msg_rseq = field_offset / 8;
        field_found++;
      }
    }

    if (field_found != 4) {
      LLDB_LOGF(GetLog(LLDBLog::Object),
                "FreeBSDKernel: Could not find all required fields for msgbuf");
      return;
    }
  } else {
    // Fallback: use hardcoded offsets based on struct layout
    // struct msgbuf layout (from sys/sys/msgbuf.h):
    //   char *msg_ptr;      - offset 0
    //   u_int msg_magic;    - offset ptr_size
    //   u_int msg_size;     - offset ptr_size + 4
    //   u_int msg_wseq;     - offset ptr_size + 8
    //   u_int msg_rseq;     - offset ptr_size + 12
    uint32_t ptr_size = GetAddressByteSize();
    offset_msg_ptr = 0;
    offset_msg_size = ptr_size + 4;
    offset_msg_wseq = ptr_size + 8;
    offset_msg_rseq = ptr_size + 12;
  }

  // Read struct msgbuf fields
  lldb::addr_t bufp = ReadPointerFromMemory(msgbufp + offset_msg_ptr, error);
  if (!error.Success() || bufp == LLDB_INVALID_ADDRESS)
    return;

  uint32_t size =
      ReadUnsignedIntegerFromMemory(msgbufp + offset_msg_size, 4, 0, error);
  if (!error.Success() || size == 0)
    return;

  uint32_t wseq =
      ReadUnsignedIntegerFromMemory(msgbufp + offset_msg_wseq, 4, 0, error);
  if (!error.Success())
    return;

  uint32_t rseq =
      ReadUnsignedIntegerFromMemory(msgbufp + offset_msg_rseq, 4, 0, error);
  if (!error.Success())
    return;

  // Convert sequences to positions
  // MSGBUF_SEQ_TO_POS macro in FreeBSD: ((seq) % (size))
  uint32_t rseq_pos = rseq % size;
  uint32_t wseq_pos = wseq % size;

  if (rseq_pos == wseq_pos)
    return;

  // Print crash info at once using stream
  lldb::StreamSP stream_sp = debugger.GetAsyncOutputStream();
  if (!stream_sp)
    return;

  stream_sp->PutCString("\nUnread portion of the kernel message buffer:\n");

  // Read ring buffer in at most two chunks
  if (rseq_pos < wseq_pos) {
    // No wrap: read from rseq_pos to wseq_pos
    size_t len = wseq_pos - rseq_pos;
    std::string buf(len, '\0');
    size_t bytes_read = ReadMemory(bufp + rseq_pos, &buf[0], len, error);
    if (error.Success() && bytes_read > 0) {
      buf.resize(bytes_read);
      *stream_sp << buf;
    }
  } else {
    // Wrap around: read from rseq_pos to end, then from start to wseq_pos
    size_t len1 = size - rseq_pos;
    std::string buf1(len1, '\0');
    size_t bytes_read1 = ReadMemory(bufp + rseq_pos, &buf1[0], len1, error);
    if (error.Success() && bytes_read1 > 0) {
      buf1.resize(bytes_read1);
      *stream_sp << buf1;
    }

    if (wseq_pos > 0) {
      std::string buf2(wseq_pos, '\0');
      size_t bytes_read2 = ReadMemory(bufp, &buf2[0], wseq_pos, error);
      if (error.Success() && bytes_read2 > 0) {
        buf2.resize(bytes_read2);
        *stream_sp << buf2;
      }
    }
  }

  stream_sp->PutChar('\n');
  stream_sp->Flush();
}

size_t ProcessFreeBSDKernel::DoReadMemory(lldb::addr_t addr, void *buf,
                                          size_t size, Status &error) {
  ssize_t rd = 0;
  rd = kvm_read2(m_kvm, addr, buf, size);
  if (rd < 0 || static_cast<size_t>(rd) != size) {
    error = Status::FromErrorStringWithFormat("Reading memory failed: %s",
                                              GetError());
    return rd > 0 ? rd : 0;
  }
  return rd;
}

const char *ProcessFreeBSDKernel::GetError() { return kvm_geterr(m_kvm); }
