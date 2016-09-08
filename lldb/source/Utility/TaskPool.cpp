//===--------------------- TaskPool.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/TaskPool.h"

namespace {
class TaskPoolImpl {
public:
  static TaskPoolImpl &GetInstance();

  void AddTask(std::function<void()> &&task_fn);

private:
  TaskPoolImpl();

  static void Worker(TaskPoolImpl *pool);

  std::queue<std::function<void()>> m_tasks;
  std::mutex m_tasks_mutex;
  uint32_t m_thread_count;
};

} // end of anonymous namespace

TaskPoolImpl &TaskPoolImpl::GetInstance() {
  static TaskPoolImpl g_task_pool_impl;
  return g_task_pool_impl;
}

void TaskPool::AddTaskImpl(std::function<void()> &&task_fn) {
  TaskPoolImpl::GetInstance().AddTask(std::move(task_fn));
}

TaskPoolImpl::TaskPoolImpl() : m_thread_count(0) {}

void TaskPoolImpl::AddTask(std::function<void()> &&task_fn) {
  static const uint32_t max_threads = std::thread::hardware_concurrency();

  std::unique_lock<std::mutex> lock(m_tasks_mutex);
  m_tasks.emplace(std::move(task_fn));
  if (m_thread_count < max_threads) {
    m_thread_count++;
    lock.unlock();

    std::thread(Worker, this).detach();
  }
}

void TaskPoolImpl::Worker(TaskPoolImpl *pool) {
  while (true) {
    std::unique_lock<std::mutex> lock(pool->m_tasks_mutex);
    if (pool->m_tasks.empty()) {
      pool->m_thread_count--;
      break;
    }

    std::function<void()> f = pool->m_tasks.front();
    pool->m_tasks.pop();
    lock.unlock();

    f();
  }
}
