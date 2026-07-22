//===------ State.cpp - Kernel Language (CUDA/HIP) persistent state -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "State.h"
#include "Types.h"

#include "OffloadAPI.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstdio>
#include <mutex>

#define CHECK_FATAL(Result, ...)                                               \
  if (Result && Result->Code) {                                                \
    fprintf(stderr, __VA_ARGS__);                                              \
    abort();                                                                   \
  }

using namespace llvm;
using namespace offload;

std::atomic<uint32_t> AnyNonDefaultDevice = 0;
__attribute__((weak)) uint32_t PerThreadQueue = 0;

thread_local CallConfigurationTy CC = {};

static std::mutex StateLock;
static std::atomic<StateTy *> StatePtr = nullptr;

static thread_local ThreadStateTy *ThreadState = nullptr;

static std::mutex ThreadStatesLock;
using ThreadStatesTy = SmallVector<ThreadStateTy *, 64>;
static ThreadStatesTy *ThreadStatesPtr = nullptr;

static void deleteThreadState() {
  std::lock_guard<std::mutex> LG(ThreadStatesLock);
  ThreadStatesTy *ThreadStates = ThreadStatesPtr;
  ThreadStatesPtr = nullptr;
  if (!ThreadStates)
    return;

  for (auto *TS : *ThreadStates)
    delete (TS);
  delete ThreadStates;
  ThreadState = nullptr;
}

static void deleteState() {
  StateTy *ST = StatePtr.load();
  StatePtr.store(nullptr);
  delete (ST);
}

static void destroyQueue(ol_queue_handle_t &Queue) {
  if (!Queue)
    return;

  olSyncQueue(Queue);
  olDestroyQueue(Queue);
  Queue = nullptr;
}

ThreadStateTy::ThreadStateTy() {
  if (PerThreadQueue) [[unlikely]]
    createDefaultQueue(getDefaultDevice());
  atexit(deleteThreadState);
}
ThreadStateTy::~ThreadStateTy() { destroyQueue(DefaultQueue); }

ThreadStateTy &ThreadStateTy::get() {
  auto *TS = ThreadState;
  if (!TS) {
    TS = new ThreadStateTy();
    ThreadState = TS;
    std::lock_guard<std::mutex> LG(ThreadStatesLock);
    if (!ThreadStatesPtr)
      ThreadStatesPtr = new ThreadStatesTy;
    ThreadStatesPtr->push_back(TS);
  }
  return *TS;
}

ol_device_handle_t ThreadStateTy::getDefaultDevice() {
  ol_device_handle_t DD = ThreadStateTy::get().DefaultDevice;
  if (DD)
    return DD;
  for (ol_device_handle_t Device : StateTy::get().getDevices()) {
    DD = Device;
    break;
  }
  if (AnyNonDefaultDevice.load(std::memory_order_relaxed)) [[unlikely]] {
    ol_device_handle_t TDD = ThreadStateTy::get().DefaultDevice;
    if (TDD)
      DD = TDD;
  }
  return DD;
}

ol_queue_handle_t ThreadStateTy::getDefaultQueue() {
  if (!PerThreadQueue) [[likely]]
    return StateTy::get().DefaultQueue;
  return ThreadStateTy::get().DefaultQueue;
}

CallConfigurationTy &ThreadStateTy::getCallConfiguration() {
  return ThreadStateTy::get().CC;
}

void ThreadStateTy::setDefaultDevice(ol_device_handle_t Device) {
  DefaultDevice = Device;
  createDefaultQueue(Device);
}

void ThreadStateTy::createDefaultQueue(ol_device_handle_t Device) {
  if (DefaultQueue)
    olDestroyQueue(DefaultQueue);
  CHECK_FATAL(olCreateQueue(Device, &DefaultQueue),
              "Failed to create per-thread default queue");
}

StateTy &StateTy::get() {
  StateTy *ST = StatePtr.load();
  if (!ST) [[unlikely]] {
    std::lock_guard<std::mutex> LG(StateLock);
    ST = StatePtr.load();
    if (!ST) {
      ST = new StateTy();
      StatePtr.store(ST);
    }
  }
  return *ST;
}

StateTy *StateTy::tryGet() { return StatePtr.load(); }

static bool addDevices(ol_device_handle_t Device, void *Payload) {
  StateTy &State = *reinterpret_cast<StateTy *>(Payload);
  ol_platform_handle_t Platform;
  ol_result_t Result;

  Result = olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                           &Platform);
  if (Result && Result->Code)
    return true;

  ol_platform_backend_t Backend;
  Result = olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                             sizeof(Backend), &Backend);
  if (Result && Result->Code)
    return true;

  if (Backend == OL_PLATFORM_BACKEND_HOST)
    State.setHostDevice(Device);
  else
    State.addDevice(Device);
  return true;
}

StateTy::StateTy() {
  CHECK_FATAL(olInit(nullptr), "Failed to initialize the LLVMOffload");
  CHECK_FATAL(olIterateDevices(addDevices, this), "Failed to identify devices");

  if (!PerThreadQueue) [[likely]]
    if (!Devices.empty()) [[likely]]
      CHECK_FATAL(olCreateQueue(Devices.front(), &DefaultQueue),
                  "Failed to create default queue");

  atexit(deleteState);
}

StateTy::~StateTy() {
  deleteThreadState();
  destroyQueue(DefaultQueue);
  destroyRegisteredPrograms();
  olShutDown();
}

void StateTy::destroyRegisteredPrograms() {
  SmallPtrSet<ol_program_handle_t, 8> Programs;
  for (auto &It : BinaryRegisterMap)
    Programs.insert(It.second);

  KernelMap.clear();
  BinaryRegisterMap.clear();

  for (ol_program_handle_t Program : Programs)
    olDestroyProgram(Program);
}
