//===-- ThreadMemory.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_THREADMEMORY_H
#define LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_THREADMEMORY_H

#include <string>

#include "lldb/Target/Thread.h"

/// A memory thread with its own ID, optionally backed by a real thread.
/// Most methods of this class dispatch to the real thread if it is not null.
/// Notable exceptions are the methods calculating the StopInfo and
/// RegisterContext of the thread, those may query the OS plugin that created
/// the thread.
class ThreadMemory : public lldb_private::Thread {
public:
  ThreadMemory(lldb_private::Process &process, lldb::tid_t tid,
               lldb::addr_t register_data_addr)
      : Thread(process, tid), m_register_data_addr(register_data_addr) {}

  ~ThreadMemory() override;

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;

  bool CalculateStopInfo() override;

  const char *GetInfo() override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetInfo();
    return nullptr;
  }

  const char *GetName() override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetName();
    return nullptr;
  }

  const char *GetQueueName() override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetQueueName();
    return nullptr;
  }

  void WillResume(lldb::StateType resume_state) override;

  void DidResume() override {
    if (m_backing_thread_sp)
      m_backing_thread_sp->DidResume();
  }

  lldb::user_id_t GetProtocolID() const override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetProtocolID();
    return Thread::GetProtocolID();
  }

  void RefreshStateAfterStop() override;

  void ClearStackFrames() override;

  void ClearBackingThread() override {
    if (m_backing_thread_sp)
      m_backing_thread_sp->ClearBackedThread();
    m_backing_thread_sp.reset();
  }

  bool SetBackingThread(const lldb::ThreadSP &thread_sp) override {
    m_backing_thread_sp = thread_sp;
    thread_sp->SetBackedThread(*this);
    return thread_sp.get();
  }

  lldb::ThreadSP GetBackingThread() const override {
    return m_backing_thread_sp;
  }

  bool IsOperatingSystemPluginThread() const override { return true; }

private:
  lldb::addr_t m_register_data_addr;
  lldb::ThreadSP m_backing_thread_sp;

  ThreadMemory(const ThreadMemory &) = delete;
  const ThreadMemory &operator=(const ThreadMemory &) = delete;
};

/// A ThreadMemory that optionally overrides the thread name.
class ThreadMemoryProvidingName : public ThreadMemory {
public:
  ThreadMemoryProvidingName(lldb_private::Process &process, lldb::tid_t tid,
                            lldb::addr_t register_data_addr,
                            llvm::StringRef name)
      : ThreadMemory(process, tid, register_data_addr), m_name(name) {}

  const char *GetName() override {
    if (!m_name.empty())
      return m_name.c_str();
    return ThreadMemory::GetName();
  }

  ~ThreadMemoryProvidingName() override = default;

private:
  std::string m_name;
};

/// A ThreadMemoryProvidingName that optionally overrides queue information.
class ThreadMemoryProvidingNameAndQueue : public ThreadMemoryProvidingName {
public:
  ThreadMemoryProvidingNameAndQueue(
      lldb_private::Process &process, lldb::tid_t tid,
      const lldb::ValueObjectSP &thread_info_valobj_sp);

  ThreadMemoryProvidingNameAndQueue(lldb_private::Process &process,
                                    lldb::tid_t tid, llvm::StringRef name,
                                    llvm::StringRef queue,
                                    lldb::addr_t register_data_addr);

  ~ThreadMemoryProvidingNameAndQueue() override = default;

  const char *GetQueueName() override {
    if (!m_queue.empty())
      return m_queue.c_str();
    return ThreadMemory::GetQueueName();
  }

  lldb::ValueObjectSP &GetValueObject() { return m_thread_info_valobj_sp; }

protected:
  lldb::ValueObjectSP m_thread_info_valobj_sp;
  std::string m_queue;

private:
  ThreadMemoryProvidingNameAndQueue(const ThreadMemoryProvidingNameAndQueue &) =
      delete;
  const ThreadMemoryProvidingNameAndQueue &
  operator=(const ThreadMemoryProvidingNameAndQueue &) = delete;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_UTILITY_THREADMEMORY_H
