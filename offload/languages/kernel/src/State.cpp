//===-- State.cpp - Kernel language persistent state ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "State.h"
#include "Stream.h"
#include "Types.h"

#include "OffloadAPI.h"
#include "OffloadErrors.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>

using namespace llvm;
using namespace offload;

// Weak fallback used unless the driver links the strong per-thread default
// stream mode object for -fgpu-default-stream=per-thread.
extern "C" {
__attribute__((weak)) uint32_t __LLVMOffloadingPerThreadDefaultStream = 0;
}

// Process-wide singleton and thread-state registry.
static std::mutex &getStateLock() {
  static std::mutex StateLock;
  return StateLock;
}
static std::atomic<StateTy *> StatePtr = nullptr;

static thread_local ThreadStateTy *ThreadState = nullptr;

static std::mutex &getThreadStatesLock() {
  static std::mutex ThreadStatesLock;
  return ThreadStatesLock;
}
using ThreadStatesTy = SmallVector<ThreadStateTy *, 64>;
static ThreadStatesTy *ThreadStatesPtr = nullptr;

static void deleteThreadStates() {
  // Detach the registry before deletion because deleteThreadState may be called
  // more than once via atexit and StateTy teardown.
  std::lock_guard<std::mutex> LG(getThreadStatesLock());
  ThreadStatesTy *ThreadStates = ThreadStatesPtr;
  ThreadStatesPtr = nullptr;
  if (!ThreadStates)
    return;

  for (auto *TS : *ThreadStates)
    delete TS;
  delete ThreadStates;
  ThreadState = nullptr;
  ThreadStates = nullptr;
}

static void deleteState() {
  StateTy *ST = StatePtr.load(std::memory_order_acquire);
  if (!ST)
    return;
  delete ST;
  StatePtr.store(nullptr, std::memory_order_release);
}

static void destroyStreamHandle(StreamTy *&Stream) {
  if (!Stream)
    return;

  olSyncQueue(Stream->Queue);
  olDestroyQueue(Stream->Queue);
  delete Stream;
  Stream = nullptr;
}

namespace llvm {
namespace offload {

static bool removeStreamFromMap(
    DenseMap<ol_device_handle_t, SmallPtrSet<StreamTy *, 8>> &StreamsMap,
    StreamTy *Stream) {
  bool Removed = false;
  SmallVector<ol_device_handle_t, 8> EmptyDevices;

  for (auto &It : StreamsMap) {
    Removed |= It.second.erase(Stream);
    if (It.second.empty())
      EmptyDevices.push_back(It.first);
  }

  for (ol_device_handle_t Device : EmptyDevices)
    StreamsMap.erase(Device);

  return Removed;
}

// ThreadStateTy implementation.

ThreadStateTy::ThreadStateTy() {
  unsigned int NumDevices = StateTy::get().Devices.size();
  PerThreadDeviceDefaultStreamMap.reserve(NumDevices);
  atexit(deleteThreadStates);
}
ThreadStateTy::~ThreadStateTy() { destroyDefaultStreams(); }

ThreadStateTy &ThreadStateTy::get() {
  auto *&TS = ThreadState;
  if (!TS) {
    TS = new ThreadStateTy();
    std::lock_guard<std::mutex> LG(getThreadStatesLock());
    if (!ThreadStatesPtr)
      ThreadStatesPtr = new ThreadStatesTy;
    ThreadStatesPtr->push_back(TS);
  }
  return *TS;
}

ol_device_handle_t ThreadStateTy::getDefaultDevice() {
  ArrayRef<ol_device_handle_t> Devices = StateTy::get().getDevices();
  int DD = DefaultDevice;
  if (DD < 0 || DD >= static_cast<int>(Devices.size()))
    return nullptr;
  return Devices[DD];
}

StreamTy *ThreadStateTy::getDefaultStream() {
  ol_device_handle_t Device = getDefaultDevice();
  if (!Device)
    return nullptr;

  if (!__LLVMOffloadingPerThreadDefaultStream) [[likely]]
    return StateTy::get().getOrCreateDefaultStream(Device);

  return getOrCreateDefaultStream(Device);
}

ol_queue_handle_t ThreadStateTy::getDefaultQueue() {
  if (StreamTy *Stream = getDefaultStream())
    return Stream->Queue;
  return nullptr;
}

CallConfigurationTy &ThreadStateTy::getCallConfiguration() { return CC; }

ol_device_handle_t ThreadStateTy::setDefaultDevice(int DeviceNo) {
  ArrayRef<ol_device_handle_t> Devices = StateTy::get().getDevices();
  if (DeviceNo < 0 || DeviceNo >= static_cast<int>(Devices.size()))
    return nullptr;
  DefaultDevice = DeviceNo;
  return Devices[DeviceNo];
}

ol_device_handle_t ThreadStateTy::getDevice(int *DeviceNo) {
  *DeviceNo = DefaultDevice;
  return getDefaultDevice();
}

uint32_t ThreadStateTy::getLastError() { return LastError; }

uint32_t ThreadStateTy::setLastError(uint32_t Error) {
  return LastError = Error;
}

StreamTy *ThreadStateTy::getOrCreateDefaultStream(ol_device_handle_t Device) {
  if (!Device)
    return nullptr;

  StateTy &State = StateTy::get();
  ol_context_handle_t Context = State.getContext();
  if (!Context)
    return nullptr;

  StreamTy *&Stream = PerThreadDeviceDefaultStreamMap[Device];
  if (!Stream) {
    ol_queue_handle_t Queue = nullptr;
    CHECK_FATAL(olCreateQueue(Context, Device, &Queue),
                "Failed to create per-thread default queue for device");
    Stream = new StreamTy{Queue, Device, QueueKind::PerThreadDefault};
    State.addStream(Stream);
  }
  return Stream;
}

void ThreadStateTy::destroyDefaultStreams() {
  for (auto &It : PerThreadDeviceDefaultStreamMap) {
    if (StateTy *State = StateTy::tryGet())
      State->removeStream(It.second);
    destroyStreamHandle(It.second);
  }
  PerThreadDeviceDefaultStreamMap.clear();
}

// StateTy implementation.

StateTy &StateTy::get() {
  StateTy *ST = StatePtr.load(std::memory_order_acquire);
  if (!ST) [[unlikely]] {
    std::lock_guard<std::mutex> LG(getStateLock());
    ST = StatePtr.load(std::memory_order_acquire);
    if (!ST) {
      ST = new StateTy();
      StatePtr.store(ST, std::memory_order_release);
    }
  }
  return *ST;
}

StateTy *StateTy::tryGet() { return StatePtr.load(std::memory_order_acquire); }

StreamTy *StateTy::getOrCreateDefaultStream(ol_device_handle_t Device) {
  if (!Device)
    return nullptr;

  ol_context_handle_t RuntimeContext = getContext();
  if (!RuntimeContext)
    return nullptr;

  std::lock_guard<std::mutex> LG(DeviceDefaultStreamsMapLock);
  StreamTy *&Stream = DeviceDefaultStreamsMap[Device];
  if (!Stream) {
    ol_queue_handle_t Queue = nullptr;
    CHECK_FATAL(olCreateQueue(RuntimeContext, Device, &Queue),
                "Failed to create default queue for device");
    Stream = new StreamTy{Queue, Device, QueueKind::LegacyDefault};
    addStream(Stream);
  }
  return Stream;
}

void StateTy::destroyDefaultStreams() {
  std::lock_guard<std::mutex> LG(DeviceDefaultStreamsMapLock);
  for (auto &It : DeviceDefaultStreamsMap) {
    removeStream(It.second);
    destroyStreamHandle(It.second);
  }
  DeviceDefaultStreamsMap.clear();
}

void StateTy::addStream(StreamTy *Stream) {
  std::lock_guard<std::mutex> LG(DeviceStreamsMapLock);
  DeviceStreamsMap[Stream->Device].insert(Stream);
}

void StateTy::removeStream(StreamTy *Stream) {
  if (!Stream)
    return;

  {
    std::lock_guard<std::mutex> LG(DeviceStreamsMapLock);
    if (!removeStreamFromMap(DeviceStreamsMap, Stream))
      return;
  }

  std::lock_guard<std::mutex> LG(DeviceBlockingStreamsMapLock);
  removeStreamFromMap(DeviceBlockingStreamsMap, Stream);
}

SmallPtrSet<StreamTy *, 8>
StateTy::getDeviceStreams(ol_device_handle_t Device) {
  std::lock_guard<std::mutex> LG(DeviceStreamsMapLock);
  auto It = DeviceStreamsMap.find(Device);
  if (It == DeviceStreamsMap.end())
    return {};
  return It->second;
}

SmallPtrSet<StreamTy *, 8>
StateTy::getBlockingStreams(ol_device_handle_t Device) {
  std::lock_guard<std::mutex> LG(DeviceBlockingStreamsMapLock);
  auto It = DeviceBlockingStreamsMap.find(Device);
  if (It == DeviceBlockingStreamsMap.end())
    return {};
  return It->second;
}

bool StateTy::hasLegacyDefaultStream(ol_device_handle_t Device) {
  std::lock_guard<std::mutex> LG(DeviceDefaultStreamsMapLock);
  auto It = DeviceDefaultStreamsMap.find(Device);
  return It != DeviceDefaultStreamsMap.end() && It->second;
}

ol_result_t StateTy::createStream(ol_device_handle_t Device, QueueKind Kind,
                                  StreamTy **Stream) {
  if (!Stream)
    return &InvalidNullPointerError;
  *Stream = nullptr;

  ol_context_handle_t RuntimeContext = getContext();
  if (!Device || !RuntimeContext)
    return &InvalidDeviceError;

  ol_queue_handle_t Queue = nullptr;
  ol_result_t Result = olCreateQueue(RuntimeContext, Device, &Queue);
  if (Result == OL_SUCCESS) {
    *Stream = new StreamTy{Queue, Device, Kind};
    addStream(*Stream);
    if (Kind == QueueKind::ExplicitBlocking) {
      std::lock_guard<std::mutex> LG(DeviceBlockingStreamsMapLock);
      DeviceBlockingStreamsMap[Device].insert(*Stream);
    }
  }
  return Result;
}

ol_result_t StateTy::destroyStream(StreamTy *Stream) {
  if (!Stream)
    return &InvalidNullPointerError;

  if (!isStreamRegistered(Stream))
    return &InvalidStreamError;

  removeStream(Stream);
  ol_result_t Result = olDestroyQueue(Stream->Queue);
  delete Stream;
  return Result;
}

bool StateTy::isStreamRegistered(StreamTy *Stream) {
  if (!Stream)
    return false;

  std::lock_guard<std::mutex> LG(DeviceStreamsMapLock);
  for (auto &It : DeviceStreamsMap)
    if (It.second.contains(Stream))
      return true;
  return false;
}

ol_device_handle_t StateTy::getHostDevice() { return HostDevice; }

ol_context_handle_t StateTy::getContext() { return Context; }

int StateTy::getDeviceCount() { return Devices.size(); }

ArrayRef<ol_device_handle_t> StateTy::getDevices() const { return Devices; }

void StateTy::addDevice(ol_device_handle_t Device) {
  Devices.push_back(Device);
}

void StateTy::setHostDevice(ol_device_handle_t Device) {
  if (!HostDevice)
    HostDevice = Device;
}

void StateTy::registerKernel(const void *ID, ol_symbol_handle_t Kernel) {
  KernelMap[ID] = Kernel;
}

void StateTy::unregisterKernel(const void *ID) { KernelMap.erase(ID); }

ol_symbol_handle_t StateTy::getKernel(const void *ID) { return KernelMap[ID]; }

void StateTy::registerProgram(const void *ID, ol_program_handle_t Program) {
  BinaryRegisterMap[ID] = Program;
}

ol_program_handle_t StateTy::unregisterProgram(const void *ID) {
  auto It = BinaryRegisterMap.find(ID);
  if (It == BinaryRegisterMap.end())
    return nullptr;
  ol_program_handle_t Program = It->second;
  BinaryRegisterMap.erase(It);
  return Program;
}

ol_program_handle_t StateTy::getProgram(const void *ID) {
  assert(BinaryRegisterMap.count(ID) && "Program not registered for binary");
  return BinaryRegisterMap[ID];
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

  if (!Devices.empty())
    CHECK_FATAL(olCreateContext(Devices.size(), Devices.data(), &Context),
                "Failed to create default context");

  unsigned int DeviceCount = Devices.size();
  DeviceDefaultStreamsMap.reserve(DeviceCount);
  DeviceStreamsMap.reserve(DeviceCount);
  DeviceBlockingStreamsMap.reserve(DeviceCount);

  atexit(deleteState);
}

StateTy::~StateTy() {
  deleteThreadStates();
  destroyDefaultStreams();
  destroyRegisteredStreams();
  destroyRegisteredPrograms();
  if (Context)
    olDestroyContext(Context);
  olShutDown();
}

void StateTy::destroyRegisteredStreams() {
  SmallVector<StreamTy *, 16> Streams;
  {
    std::lock_guard<std::mutex> LG(DeviceStreamsMapLock);
    for (auto &It : DeviceStreamsMap)
      Streams.append(It.second.begin(), It.second.end());
    DeviceStreamsMap.clear();
  }
  {
    std::lock_guard<std::mutex> LG(DeviceBlockingStreamsMapLock);
    DeviceBlockingStreamsMap.clear();
  }

  for (StreamTy *&Stream : Streams)
    destroyStreamHandle(Stream);
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
