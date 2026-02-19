//===-- ProcessFreeBSDKernel.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/DynamicLoader.h"

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

void ProcessFreeBSDKernel::RefreshStateAfterStop() {}

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
