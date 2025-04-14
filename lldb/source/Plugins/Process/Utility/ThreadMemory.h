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

  void SetQueueName(const char *name) override {
    if (m_backing_thread_sp)
      m_backing_thread_sp->SetQueueName(name);
  }

  lldb::queue_id_t GetQueueID() override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetQueueID();
    return LLDB_INVALID_QUEUE_ID;
  }

  void SetQueueID(lldb::queue_id_t new_val) override {
    if (m_backing_thread_sp)
      m_backing_thread_sp->SetQueueID(new_val);
  }

  lldb::QueueKind GetQueueKind() override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetQueueKind();
    return lldb::eQueueKindUnknown;
  }

  void SetQueueKind(lldb::QueueKind kind) override {
    if (m_backing_thread_sp)
      m_backing_thread_sp->SetQueueKind(kind);
  }

  lldb::QueueSP GetQueue() override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetQueue();
    return lldb::QueueSP();
  }

  lldb::addr_t GetQueueLibdispatchQueueAddress() override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetQueueLibdispatchQueueAddress();
    return LLDB_INVALID_ADDRESS;
  }

  void SetQueueLibdispatchQueueAddress(lldb::addr_t dispatch_queue_t) override {
    if (m_backing_thread_sp)
      m_backing_thread_sp->SetQueueLibdispatchQueueAddress(dispatch_queue_t);
  }

  lldb_private::LazyBool GetAssociatedWithLibdispatchQueue() override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->GetAssociatedWithLibdispatchQueue();
    return lldb_private::eLazyBoolNo;
  }

  void SetAssociatedWithLibdispatchQueue(
      lldb_private::LazyBool associated_with_libdispatch_queue) override {
    if (m_backing_thread_sp)
      m_backing_thread_sp->SetAssociatedWithLibdispatchQueue(
          associated_with_libdispatch_queue);
  }

  bool ThreadHasQueueInformation() const override {
    if (m_backing_thread_sp)
      return m_backing_thread_sp->ThreadHasQueueInformation();
    return false;
  }

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

  /// TODO: this method should take into account the queue override.
  void SetQueueName(const char *name) override { Thread::SetQueueName(name); }

  /// TODO: this method should take into account the queue override.
  lldb::queue_id_t GetQueueID() override { return Thread::GetQueueID(); }

  /// TODO: this method should take into account the queue override.
  void SetQueueID(lldb::queue_id_t new_val) override {
    Thread::SetQueueID(new_val);
  }

  /// TODO: this method should take into account the queue override.
  lldb::QueueKind GetQueueKind() override { return Thread::GetQueueKind(); }

  /// TODO: this method should take into account the queue override.
  void SetQueueKind(lldb::QueueKind kind) override {
    Thread::SetQueueKind(kind);
  }

  /// TODO: this method should take into account the queue override.
  lldb::QueueSP GetQueue() override { return Thread::GetQueue(); }

  /// TODO: this method should take into account the queue override.
  lldb::addr_t GetQueueLibdispatchQueueAddress() override {
    return Thread::GetQueueLibdispatchQueueAddress();
  }

  /// TODO: this method should take into account the queue override.
  void SetQueueLibdispatchQueueAddress(lldb::addr_t dispatch_queue_t) override {
    Thread::SetQueueLibdispatchQueueAddress(dispatch_queue_t);
  }

  /// TODO: this method should take into account the queue override.
  bool ThreadHasQueueInformation() const override {
    return Thread::ThreadHasQueueInformation();
  }

  /// TODO: this method should take into account the queue override.
  lldb_private::LazyBool GetAssociatedWithLibdispatchQueue() override {
    return Thread::GetAssociatedWithLibdispatchQueue();
  }

  /// TODO: this method should take into account the queue override.
  void SetAssociatedWithLibdispatchQueue(
      lldb_private::LazyBool associated_with_libdispatch_queue) override {
    Thread::SetAssociatedWithLibdispatchQueue(
        associated_with_libdispatch_queue);
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
