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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <mutex>

using namespace llvm;
using namespace offload;

// Weak fallback used unless the driver links the strong per-thread default
// stream mode object for -fgpu-default-stream=per-thread.
extern "C" {
__attribute__((weak)) uint32_t LLVMOffloadingPerThreadDefaultStream = 0;
}

static constexpr ol_error_struct_t InvalidNullPointerError = {
    OL_ERRC_INVALID_NULL_POINTER, "invalid null stream pointer"};

static constexpr ol_error_struct_t InvalidDeviceError = {OL_ERRC_INVALID_DEVICE,
                                                         "invalid device"};

static constexpr ol_error_struct_t InvalidStreamError = {OL_ERRC_INVALID_QUEUE,
                                                         "invalid stream"};

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

static std::mutex &getDeviceDefaultStreamsMapLock() {
  static std::mutex DeviceDefaultStreamsMapLock;
  return DeviceDefaultStreamsMapLock;
}

static std::mutex &getDeviceStreamsMapLock() {
  static std::mutex DeviceStreamsMapLock;
  return DeviceStreamsMapLock;
}

static std::mutex &getDeviceBlockingStreamsMapLock() {
  static std::mutex DeviceBlockingStreamsMapLock;
  return DeviceBlockingStreamsMapLock;
}

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
  StateTy *ST = StatePtr.load();
  if (!ST)
    return;
  delete ST;
  StatePtr.store(nullptr);
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

ThreadStateTy::ThreadStateTy() { atexit(deleteThreadStates); }
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
  int DD = ThreadStateTy::get().DefaultDevice;
  if (DD < 0 || DD >= static_cast<int>(Devices.size()))
    return nullptr;
  return Devices[DD];
}

StreamTy *ThreadStateTy::getDefaultStream() {
  ol_device_handle_t Device = getDefaultDevice();
  if (!Device)
    return nullptr;

  if (!LLVMOffloadingPerThreadDefaultStream) [[likely]]
    return StateTy::get().getOrCreateDefaultStream(Device);

  return ThreadStateTy::get().getOrCreateDefaultStream(Device);
}

ol_queue_handle_t ThreadStateTy::getDefaultQueue() {
  if (StreamTy *Stream = getDefaultStream())
    return Stream->Queue;
  return nullptr;
}

CallConfigurationTy &ThreadStateTy::getCallConfiguration() {
  return ThreadStateTy::get().CC;
}

ol_device_handle_t ThreadStateTy::setDefaultDevice(int DeviceNo) {
  ArrayRef<ol_device_handle_t> Devices = StateTy::get().getDevices();
  if (DeviceNo < 0 || DeviceNo >= static_cast<int>(Devices.size()))
    return nullptr;
  ThreadStateTy::get().DefaultDevice = DeviceNo;
  return Devices[DeviceNo];
}

ol_device_handle_t ThreadStateTy::getDevice(int *DeviceNo) {
  *DeviceNo = ThreadStateTy::get().DefaultDevice;
  return ThreadStateTy::getDefaultDevice();
}

uint32_t ThreadStateTy::getLastError() {
  return ThreadStateTy::get().LastError;
}

uint32_t ThreadStateTy::setLastError(uint32_t Error) {
  return ThreadStateTy::get().LastError = Error;
}

StreamTy *ThreadStateTy::getOrCreateDefaultStream(ol_device_handle_t Device) {
  if (!Device)
    return nullptr;

  ol_context_handle_t Context = StateTy::getContext();
  if (!Context)
    return nullptr;

  StreamTy *&Stream = PerThreadDeviceDefaultStreamMap[Device];
  if (!Stream) {
    ol_queue_handle_t Queue = nullptr;
    CHECK_FATAL(olCreateQueue(Context, Device, &Queue),
                "Failed to create per-thread default queue for device");
    Stream = new StreamTy{Queue, Device, QueueKind::PerThreadDefault};
    StateTy::get().addStream(Stream);
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
  StateTy *ST = StatePtr.load();
  if (!ST) [[unlikely]] {
    std::lock_guard<std::mutex> LG(getStateLock());
    ST = StatePtr.load();
    if (!ST) {
      ST = new StateTy();
      StatePtr.store(ST);
    }
  }
  return *ST;
}

StateTy *StateTy::tryGet() { return StatePtr.load(); }

StreamTy *StateTy::getOrCreateDefaultStream(ol_device_handle_t Device) {
  if (!Device)
    return nullptr;

  ol_context_handle_t RuntimeContext = StateTy::getContext();
  if (!RuntimeContext)
    return nullptr;

  std::lock_guard<std::mutex> LG(getDeviceDefaultStreamsMapLock());
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
  std::lock_guard<std::mutex> LG(getDeviceDefaultStreamsMapLock());
  for (auto &It : DeviceDefaultStreamsMap) {
    removeStream(It.second);
    destroyStreamHandle(It.second);
  }
  DeviceDefaultStreamsMap.clear();
}

void StateTy::addStream(StreamTy *Stream) {
  std::lock_guard<std::mutex> LG(getDeviceStreamsMapLock());
  DeviceStreamsMap[Stream->Device].insert(Stream);
}

void StateTy::removeStream(StreamTy *Stream) {
  if (!Stream)
    return;

  {
    std::lock_guard<std::mutex> LG(getDeviceStreamsMapLock());
    if (!removeStreamFromMap(DeviceStreamsMap, Stream))
      return;
  }

  std::lock_guard<std::mutex> LG(getDeviceBlockingStreamsMapLock());
  removeStreamFromMap(DeviceBlockingStreamsMap, Stream);
}

SmallPtrSet<StreamTy *, 8>
StateTy::getDeviceStreams(ol_device_handle_t Device) {
  StateTy &State = get();
  std::lock_guard<std::mutex> LG(getDeviceStreamsMapLock());
  auto It = State.DeviceStreamsMap.find(Device);
  if (It == State.DeviceStreamsMap.end())
    return {};
  return It->second;
}

SmallPtrSet<StreamTy *, 8>
StateTy::getBlockingStreams(ol_device_handle_t Device) {
  StateTy &State = get();
  std::lock_guard<std::mutex> LG(getDeviceBlockingStreamsMapLock());
  auto It = State.DeviceBlockingStreamsMap.find(Device);
  if (It == State.DeviceBlockingStreamsMap.end())
    return {};
  return It->second;
}

bool StateTy::hasLegacyDefaultStream(ol_device_handle_t Device) {
  StateTy &State = get();
  std::lock_guard<std::mutex> LG(getDeviceDefaultStreamsMapLock());
  auto It = State.DeviceDefaultStreamsMap.find(Device);
  return It != State.DeviceDefaultStreamsMap.end() && It->second;
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
    get().addStream(*Stream);
    if (Kind == QueueKind::ExplicitBlocking) {
      StateTy &State = get();
      std::lock_guard<std::mutex> LG(getDeviceBlockingStreamsMapLock());
      State.DeviceBlockingStreamsMap[Device].insert(*Stream);
    }
  }
  return Result;
}

ol_result_t StateTy::destroyStream(StreamTy *Stream) {
  if (!Stream)
    return &InvalidNullPointerError;

  if (!isStreamRegistered(Stream))
    return &InvalidStreamError;

  get().removeStream(Stream);
  ol_result_t Result = olDestroyQueue(Stream->Queue);
  delete Stream;
  return Result;
}

bool StateTy::isStreamRegistered(StreamTy *Stream) {
  if (!Stream)
    return false;

  StateTy &State = get();
  std::lock_guard<std::mutex> LG(getDeviceStreamsMapLock());
  for (auto &It : State.DeviceStreamsMap)
    if (It.second.contains(Stream))
      return true;
  return false;
}

ol_device_handle_t StateTy::getHostDevice() { return get().HostDevice; }

ol_context_handle_t StateTy::getContext() { return get().Context; }

int StateTy::getDeviceCount() {
  int DeviceCount = get().getDevices().size();
  return DeviceCount;
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

  if (!Devices.empty())
    CHECK_FATAL(olCreateContext(Devices.size(), Devices.data(), &Context),
                "Failed to create default context");

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
    std::lock_guard<std::mutex> LG(getDeviceStreamsMapLock());
    for (auto &It : DeviceStreamsMap)
      Streams.append(It.second.begin(), It.second.end());
    DeviceStreamsMap.clear();
  }
  {
    std::lock_guard<std::mutex> LG(getDeviceBlockingStreamsMapLock());
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
