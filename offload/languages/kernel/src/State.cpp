//===-- State.cpp - Kernel language persistent state ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "State.h"
#include "Types.h"

#include "OffloadAPI.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <mutex>

#define CHECK_FATAL(Result, ...)                                               \
  if (Result && Result->Code) {                                                \
    llvm::errs() << __VA_ARGS__ << '\n';                                       \
    abort();                                                                   \
  }

using namespace llvm;
using namespace offload;

// Weak so another runtime object can override the default stream mode.
__attribute__((weak)) uint32_t PerThreadQueue = 0;

thread_local CallConfigurationTy CC = {};

// Process-wide singleton and thread-state registry.
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
    delete TS;
  delete ThreadStates;
  ThreadState = nullptr;
}

static void deleteState() {
  StateTy *ST = StatePtr.load();
  StatePtr.store(nullptr);
  delete ST;
}

static void destroyQueue(ol_queue_handle_t &Queue) {
  if (!Queue)
    return;

  olSyncQueue(Queue);
  olDestroyQueue(Queue);
  Queue = nullptr;
}

namespace llvm {
namespace offload {

// ThreadStateTy implementation.

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
  ThreadStateTy &State = get();
  State.DefaultDevice = Device;
  State.createDefaultQueue(Device);
}

void ThreadStateTy::createDefaultQueue(ol_device_handle_t Device) {
  if (DefaultQueue)
    olDestroyQueue(DefaultQueue);
  CHECK_FATAL(olCreateQueue(Device, &DefaultQueue),
              "Failed to create per-thread default queue");
}

// StateTy implementation.

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

ol_device_handle_t StateTy::getHostDevice() { return get().HostDevice; }

int StateTy::getDeviceCount() {
  int DeviceCount = get().getDevices().size();
  return DeviceCount;
}

ol_device_handle_t StateTy::getDevice(int *DeviceNo) {
  ol_device_handle_t DefaultDevice = ThreadStateTy::getDefaultDevice();
  int DeviceCount = get().getDevices().size();
  ArrayRef<ol_device_handle_t> Devices = get().getDevices();
  for (int i = 0; i < DeviceCount; i++) {
    if (Devices[i] == DefaultDevice) {
      *DeviceNo = i;
      return Devices[i];
    }
  }
  return nullptr;
}

ol_device_handle_t StateTy::setDefaultDevice(int DeviceNo) {
  ArrayRef<ol_device_handle_t> Devices = get().getDevices();
  if (DeviceNo < 0 || DeviceNo >= static_cast<int>(Devices.size()))
    return nullptr;
  ol_device_handle_t Device = Devices[DeviceNo];
  ThreadStateTy::setDefaultDevice(Device);
  return Device;
}

ArrayRef<ol_device_handle_t> StateTy::getDevices() const { return Devices; }

void StateTy::addDevice(ol_device_handle_t Device) {
  Devices.push_back(Device);
}

void StateTy::setHostDevice(ol_device_handle_t Device) {
  if (!HostDevice)
    HostDevice = Device;
}

void StateTy::addKernel(KernelIDTy KernelID, ol_symbol_handle_t Kernel) {
  KernelMap[KernelID] = Kernel;
}

void StateTy::removeKernel(KernelIDTy KernelID) { KernelMap.erase(KernelID); }

ol_symbol_handle_t StateTy::lookupKernel(KernelIDTy KernelID) {
  return KernelMap[KernelID];
}

void StateTy::registerKernel(const void *ID, ol_symbol_handle_t Kernel) {
  get().addKernel(ID, Kernel);
}

void StateTy::unregisterKernel(const void *ID) {
  if (StateTy *State = tryGet())
    State->removeKernel(ID);
}

ol_symbol_handle_t StateTy::getKernel(const void *ID) {
  return get().lookupKernel(ID);
}

void StateTy::addProgram(const void *Binary, ol_program_handle_t Program) {
  BinaryRegisterMap[Binary] = Program;
}

ol_program_handle_t StateTy::removeProgram(const void *Binary) {
  auto It = BinaryRegisterMap.find(Binary);
  if (It == BinaryRegisterMap.end())
    return nullptr;
  ol_program_handle_t Program = It->second;
  BinaryRegisterMap.erase(It);
  return Program;
}

ol_program_handle_t StateTy::lookupProgram(const void *Binary) {
  assert(BinaryRegisterMap.count(Binary) &&
         "Program not registered for binary");
  return BinaryRegisterMap[Binary];
}

void StateTy::registerProgram(const void *ID, ol_program_handle_t Program) {
  get().addProgram(ID, Program);
}

ol_program_handle_t StateTy::unregisterProgram(const void *ID) {
  if (StateTy *State = tryGet())
    return State->removeProgram(ID);
  return nullptr;
}

ol_program_handle_t StateTy::getProgram(const void *ID) {
  return get().lookupProgram(ID);
}

bool StateTy::addDevices(ol_device_handle_t Device, void *Payload) {
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
  CHECK_FATAL(olIterateDevices(StateTy::addDevices, this),
              "Failed to identify devices");

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

} // namespace offload
} // namespace llvm
