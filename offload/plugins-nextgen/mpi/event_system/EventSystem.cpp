//===------ event_system.cpp - Concurrent MPI communication -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the MPI Event System used by the MPI
// target runtime for concurrent communication.
//
//===----------------------------------------------------------------------===//

#include "EventSystem.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>

#include <mpi.h>
#include <unistd.h>

#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/EnvironmentVar.h"
#include "Shared/Utils.h"
#include "omptarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include "llvm/Support/DynamicLibrary.h"

#define CHECK(expr, msg, ...)                                                  \
  if (!(expr)) {                                                               \
    REPORT(msg, ##__VA_ARGS__);                                                \
    return false;                                                              \
  }

std::string EventTypeToString(EventTypeTy eventType) {
  switch (eventType) {
  case EventTypeTy::RETRIEVE_NUM_DEVICES:
    return "RETRIEVE_NUM_DEVICES";
  case EventTypeTy::INIT_DEVICE:
    return "INIT_DEVICE";
  case EventTypeTy::INIT_RECORD_REPLAY:
    return "INIT_RECORD_REPLAY";
  case EventTypeTy::IS_PLUGIN_COMPATIBLE:
    return "IS_PLUGIN_COMPATIBLE";
  case EventTypeTy::IS_DEVICE_COMPATIBLE:
    return "IS_DEVICE_COMPATIBLE";
  case EventTypeTy::IS_DATA_EXCHANGABLE:
    return "IS_DATA_EXCHANGABLE";
  case EventTypeTy::LOAD_BINARY:
    return "LOAD_BINARY";
  case EventTypeTy::GET_GLOBAL:
    return "GET_GLOBAL";
  case EventTypeTy::GET_FUNCTION:
    return "GET_FUNCTION";
  case EventTypeTy::SYNCHRONIZE:
    return "SYNCHRONIZE";
  case EventTypeTy::INIT_ASYNC_INFO:
    return "INIT_ASYNC_INFO";
  case EventTypeTy::INIT_DEVICE_INFO:
    return "INIT_DEVICE_INFO";
  case EventTypeTy::QUERY_ASYNC:
    return "QUERY_ASYNC";
  case EventTypeTy::PRINT_DEVICE_INFO:
    return "PRINT_DEVICE_INFO";
  case EventTypeTy::DATA_LOCK:
    return "DATA_LOCK";
  case EventTypeTy::DATA_UNLOCK:
    return "DATA_UNLOCK";
  case EventTypeTy::DATA_NOTIFY_MAPPED:
    return "DATA_NOTIFY_MAPPED";
  case EventTypeTy::DATA_NOTIFY_UNMAPPED:
    return "DATA_NOTIFY_UNMAPPED";
  case EventTypeTy::ALLOC:
    return "ALLOC";
  case EventTypeTy::DELETE:
    return "DELETE";
  case EventTypeTy::SUBMIT:
    return "SUBMIT";
  case EventTypeTy::RETRIEVE:
    return "RETRIEVE";
  case EventTypeTy::LOCAL_EXCHANGE:
    return "LOCAL_EXCHANGE";
  case EventTypeTy::EXCHANGE_SRC:
    return "EXCHANGE_SRC";
  case EventTypeTy::EXCHANGE_DST:
    return "EXCHANGE_DST";
  case EventTypeTy::LAUNCH_KERNEL:
    return "LAUNCH_KERNEL";
  case EventTypeTy::SYNC:
    return "SYNC";
  case EventTypeTy::EXIT:
    return "EXIT";
  default:
    return "UNKNOWN_EVENT_TYPE";
  }
}

/// Resumes the most recent incomplete coroutine in the list.
void EventTy::resume() {
  // Acquire first handle not done.
  const CoHandleTy &RootHandle = getHandle().promise().RootHandle;
  auto &ResumableHandle = RootHandle.promise().PrevHandle;
  while (ResumableHandle.done()) {
    ResumableHandle = ResumableHandle.promise().PrevHandle;

    if (ResumableHandle == RootHandle)
      break;
  }

  if (!ResumableHandle.done())
    ResumableHandle.resume();
}

/// Wait until event completes.
void EventTy::wait() {
  // Advance the event progress until it is completed.
  while (!done()) {
    advance();
  }
}

/// Advance the event to the next suspension point and wait a while
void EventTy::advance() {
  resume();
  std::this_thread::sleep_for(
      std::chrono::microseconds(EventPollingRate.get()));
}

/// Check if the event has completed.
bool EventTy::done() const { return getHandle().done(); }

/// Check if it is an empty event.
bool EventTy::empty() const { return !getHandle(); }

/// Get the coroutine error from the Handle.
llvm::Error EventTy::getError() const {
  auto &Error = getHandle().promise().CoroutineError;
  if (Error)
    return std::move(*Error);

  return llvm::Error::success();
}

///  MPI Request Manager Destructor. The Manager cannot be destroyed until all
///  the requests it manages have been completed.
MPIRequestManagerTy::~MPIRequestManagerTy() {
  assert(Requests.empty() && "Requests must be fulfilled and emptied before "
                             "destruction. Did you co_await on it?");
}

/// Send a message to \p OtherRank asynchronously.
void MPIRequestManagerTy::send(const void *Buffer, int Size,
                               MPI_Datatype Datatype) {
  MPI_Isend(Buffer, Size, Datatype, OtherRank, Tag, Comm,
            &Requests.emplace_back(MPI_REQUEST_NULL));
}

/// Divide the \p Buffer into fragments of size \p MPIFragmentSize and send them
/// to \p OtherRank asynchronously.
void MPIRequestManagerTy::sendInBatchs(void *Buffer, int64_t Size) {
  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  char *BufferByteArray = reinterpret_cast<char *>(Buffer);
  int64_t RemainingBytes = Size;
  while (RemainingBytes > 0) {
    send(&BufferByteArray[Size - RemainingBytes],
         static_cast<int>(std::min(RemainingBytes, MPIFragmentSize.get())),
         MPI_BYTE);
    RemainingBytes -= MPIFragmentSize.get();
  }
}

/// Receive a message from \p OtherRank asynchronously.
void MPIRequestManagerTy::receive(void *Buffer, int Size,
                                  MPI_Datatype Datatype) {
  MPI_Irecv(Buffer, Size, Datatype, OtherRank, Tag, Comm,
            &Requests.emplace_back(MPI_REQUEST_NULL));
}

/// Asynchronously receive message fragments from \p OtherRank and reconstruct
/// them into \p Buffer.
void MPIRequestManagerTy::receiveInBatchs(void *Buffer, int64_t Size) {
  // Operates over many fragments of the original buffer of at most
  // MPI_FRAGMENT_SIZE bytes.
  char *BufferByteArray = reinterpret_cast<char *>(Buffer);
  int64_t RemainingBytes = Size;
  while (RemainingBytes > 0) {
    receive(&BufferByteArray[Size - RemainingBytes],
            static_cast<int>(std::min(RemainingBytes, MPIFragmentSize.get())),
            MPI_BYTE);
    RemainingBytes -= MPIFragmentSize.get();
  }
}

/// Coroutine that waits until all pending requests finish.
EventTy MPIRequestManagerTy::wait() {
  int RequestsCompleted = false;

  while (!RequestsCompleted) {
    int MPIError = MPI_Testall(Requests.size(), Requests.data(),
                               &RequestsCompleted, MPI_STATUSES_IGNORE);

    if (MPIError != MPI_SUCCESS)
      co_return createError("Waiting of MPI requests failed with code %d",
                            MPIError);

    co_await std::suspend_always{};
  }

  Requests.clear();

  co_return llvm::Error::success();
}

EventTy operator co_await(MPIRequestManagerTy &RequestManager) {
  return RequestManager.wait();
}

void *memAllocHost(int64_t Size) {
  void *HstPtr = nullptr;
  int MPIError = MPI_Alloc_mem(Size, MPI_INFO_NULL, &HstPtr);
  if (MPIError != MPI_SUCCESS)
    return nullptr;
  return HstPtr;
}

int memFreeHost(void *HstPtr) {
  int MPIError = MPI_Free_mem(HstPtr);
  if (MPIError != MPI_SUCCESS)
    return OFFLOAD_FAIL;
  return OFFLOAD_SUCCESS;
}

/// Event implementations on Host side.
namespace OriginEvents {

EventTy retrieveNumDevices(MPIRequestManagerTy RequestManager,
                           int32_t *NumDevices) {
  RequestManager.receive(NumDevices, 1, MPI_INT32_T);
  co_return (co_await RequestManager);
}

EventTy isPluginCompatible(MPIRequestManagerTy RequestManager,
                           __tgt_device_image *Image, bool *QueryResult) {
  uint64_t Size = utils::getPtrDiff(Image->ImageEnd, Image->ImageStart);

  void *Buffer = memAllocHost(Size);
  if (Buffer != nullptr)
    memcpy(Buffer, Image->ImageStart, Size);
  else
    Buffer = Image->ImageStart;

  RequestManager.send(&Size, 1, MPI_UINT64_T);
  RequestManager.send(Buffer, Size, MPI_BYTE);

  if (auto Err = co_await RequestManager; Err)
    co_return Err;

  if (Buffer != Image->ImageStart)
    memFreeHost(Buffer);

  RequestManager.receive(QueryResult, sizeof(bool), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy isDeviceCompatible(MPIRequestManagerTy RequestManager,
                           __tgt_device_image *Image, bool *QueryResult) {
  uint64_t Size = utils::getPtrDiff(Image->ImageEnd, Image->ImageStart);

  void *Buffer = memAllocHost(Size);
  if (Buffer != nullptr)
    memcpy(Buffer, Image->ImageStart, Size);
  else
    Buffer = Image->ImageStart;

  RequestManager.send(&Size, 1, MPI_UINT64_T);
  RequestManager.send(Buffer, Size, MPI_BYTE);

  if (auto Err = co_await RequestManager; Err)
    co_return Err;

  if (Buffer != Image->ImageStart)
    memFreeHost(Buffer);

  RequestManager.receive(QueryResult, sizeof(bool), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy initDevice(MPIRequestManagerTy RequestManager, void **DevicePtr) {
  // Wait the complete notification
  RequestManager.receive(DevicePtr, sizeof(void *), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy initRecordReplay(MPIRequestManagerTy RequestManager, int64_t MemorySize,
                         void *VAddr, bool IsRecord, bool SaveOutput,
                         uint64_t *ReqPtrArgOffset) {
  RequestManager.send(&MemorySize, 1, MPI_INT64_T);
  RequestManager.send(&VAddr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&IsRecord, sizeof(bool), MPI_BYTE);
  RequestManager.send(&SaveOutput, sizeof(bool), MPI_BYTE);
  RequestManager.receive(&ReqPtrArgOffset, 1, MPI_UINT64_T);
  co_return (co_await RequestManager);
}

EventTy isDataExchangable(MPIRequestManagerTy RequestManager,
                          int32_t DstDeviceId, bool *QueryResult) {
  RequestManager.send(&DstDeviceId, 1, MPI_INT32_T);
  RequestManager.receive(QueryResult, sizeof(bool), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy allocateBuffer(MPIRequestManagerTy RequestManager, int64_t Size,
                       int32_t Kind, void **Buffer) {
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&Kind, 1, MPI_INT32_T);

  // Event completion notification
  RequestManager.receive(Buffer, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy deleteBuffer(MPIRequestManagerTy RequestManager, void *Buffer,
                     int32_t Kind) {
  RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Kind, 1, MPI_INT32_T);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy submit(MPIRequestManagerTy RequestManager, void *TgtPtr, void *HstPtr,
               int64_t Size, __tgt_async_info *AsyncInfoPtr) {
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  RequestManager.send(&TgtPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);

  RequestManager.sendInBatchs(HstPtr, Size);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy retrieve(MPIRequestManagerTy RequestManager, int64_t Size, void *HstPtr,
                 void *TgtPtr, __tgt_async_info *AsyncInfoPtr) {
  bool DeviceOpStatus = true;

  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  RequestManager.send(&TgtPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);

  RequestManager.receive(&DeviceOpStatus, sizeof(bool), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  if (!DeviceOpStatus)
    co_return createError("Failed to retrieve %p TgtPtr.", TgtPtr);

  RequestManager.receiveInBatchs(HstPtr, Size);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy localExchange(MPIRequestManagerTy RequestManager, void *SrcPtr,
                      int DstDeviceId, void *DstPtr, int64_t Size,
                      __tgt_async_info *AsyncInfoPtr) {
  RequestManager.send(&SrcPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&DstDeviceId, 1, MPI_INT);
  RequestManager.send(&DstPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy exchange(MPIRequestManagerTy RequestManager, int SrcRank,
                 const void *OrgBuffer, int DstRank, void *DstBuffer,
                 int64_t Size, __tgt_async_info *AsyncInfoPtr) {
  // Send data to SrcRank
  RequestManager.send(&OrgBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&DstRank, 1, MPI_INT);
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  // Send data to DstRank
  RequestManager.OtherRank = DstRank;
  RequestManager.send(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&SrcRank, 1, MPI_INT);
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  RequestManager.OtherRank = SrcRank;
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy launchKernel(MPIRequestManagerTy RequestManager, void *TgtEntryPtr,
                     EventDataHandleTy TgtArgs, EventDataHandleTy TgtOffsets,
                     EventDataHandleTy KernelArgsHandle,
                     __tgt_async_info *AsyncInfoPtr) {
  KernelArgsTy *KernelArgs =
      static_cast<KernelArgsTy *>(KernelArgsHandle.get());

  RequestManager.send(&KernelArgs->NumArgs, 1, MPI_UINT32_T);
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&TgtEntryPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(TgtArgs.get(), KernelArgs->NumArgs * sizeof(void *),
                      MPI_BYTE);
  RequestManager.send(TgtOffsets.get(), KernelArgs->NumArgs * sizeof(ptrdiff_t),
                      MPI_BYTE);

  RequestManager.send(KernelArgs, sizeof(KernelArgsTy), MPI_BYTE);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy getGlobal(MPIRequestManagerTy RequestManager,
                  __tgt_device_binary Binary, uint64_t Size, const char *Name,
                  void **DevicePtr) {
  uint32_t NameSize = strlen(Name) + 1;
  RequestManager.send(&Binary.handle, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_UINT64_T);
  RequestManager.send(&NameSize, 1, MPI_UINT32_T);
  RequestManager.send(Name, NameSize, MPI_CHAR);

  RequestManager.receive(DevicePtr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy getFunction(MPIRequestManagerTy RequestManager,
                    __tgt_device_binary Binary, const char *Name,
                    void **KernelPtr) {
  RequestManager.send(&Binary.handle, sizeof(void *), MPI_BYTE);
  uint32_t Size = strlen(Name) + 1;
  RequestManager.send(&Size, 1, MPI_UINT32_T);
  RequestManager.send(Name, Size, MPI_CHAR);

  RequestManager.receive(KernelPtr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy synchronize(MPIRequestManagerTy RequestManager,
                    __tgt_async_info *AsyncInfoPtr) {
  bool DeviceOpStatus = false;
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  RequestManager.receive(&DeviceOpStatus, sizeof(bool), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  if (!DeviceOpStatus)
    co_return createError("Failed to synchronize device.");

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy sync(EventTy Event) {
  while (!Event.done())
    co_await std::suspend_always{};

  co_return llvm::Error::success();
}

EventTy loadBinary(MPIRequestManagerTy RequestManager,
                   const __tgt_device_image *Image,
                   __tgt_device_binary *Binary) {
  auto &[ImageStart, ImageEnd, EntriesBegin, EntriesEnd] = *Image;

  // Send the target table sizes.
  size_t ImageSize = (size_t)ImageEnd - (size_t)ImageStart;
  size_t EntryCount = EntriesEnd - EntriesBegin;
  llvm::SmallVector<size_t> EntryNameSizes(EntryCount);

  for (size_t I = 0; I < EntryCount; I++) {
    // Note: +1 for the terminator.
    EntryNameSizes[I] = std::strlen(EntriesBegin[I].SymbolName) + 1;
  }

  RequestManager.send(&ImageSize, 1, MPI_UINT64_T);
  RequestManager.send(&EntryCount, 1, MPI_UINT64_T);
  RequestManager.send(EntryNameSizes.begin(), EntryCount, MPI_UINT64_T);

  void *Buffer = memAllocHost(ImageSize);
  if (Buffer != nullptr)
    memcpy(Buffer, ImageStart, ImageSize);
  else
    Buffer = ImageStart;

  // Send the image bytes and the table entries.
  RequestManager.send(Buffer, ImageSize, MPI_BYTE);

  for (size_t I = 0; I < EntryCount; I++) {
    RequestManager.send(&EntriesBegin[I].Reserved, 1, MPI_UINT64_T);
    RequestManager.send(&EntriesBegin[I].Version, 1, MPI_UINT16_T);
    RequestManager.send(&EntriesBegin[I].Kind, 1, MPI_UINT16_T);
    RequestManager.send(&EntriesBegin[I].Flags, 1, MPI_INT32_T);
    RequestManager.send(&EntriesBegin[I].Address, 1, MPI_UINT64_T);
    RequestManager.send(EntriesBegin[I].SymbolName, EntryNameSizes[I],
                        MPI_CHAR);
    RequestManager.send(&EntriesBegin[I].Size, 1, MPI_UINT64_T);
    RequestManager.send(&EntriesBegin[I].Data, 1, MPI_INT32_T);
    RequestManager.send(&EntriesBegin[I].AuxAddr, 1, MPI_UINT64_T);
  }

  if (auto Err = co_await RequestManager; Err)
    co_return Err;

  if (Buffer != ImageStart)
    memFreeHost(Buffer);

  RequestManager.receive(&Binary->handle, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy queryAsync(MPIRequestManagerTy RequestManager,
                   __tgt_async_info *AsyncInfoPtr) {
  RequestManager.send(&AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  if (auto Err = co_await RequestManager; Err)
    co_return Err;

  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy printDeviceInfo(MPIRequestManagerTy RequestManager) {
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy initAsyncInfo(MPIRequestManagerTy RequestManager,
                      __tgt_async_info **AsyncInfoPtr) {
  RequestManager.receive(AsyncInfoPtr, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy initDeviceInfo(MPIRequestManagerTy RequestManager,
                       __tgt_device_info *DeviceInfo) {
  RequestManager.send(DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receive(DeviceInfo, sizeof(__tgt_device_info), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy dataLock(MPIRequestManagerTy RequestManager, void *Ptr, int64_t Size,
                 void **LockedPtr) {
  RequestManager.send(&Ptr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.receive(LockedPtr, sizeof(void *), MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy dataUnlock(MPIRequestManagerTy RequestManager, void *Ptr) {
  RequestManager.send(&Ptr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy dataNotifyMapped(MPIRequestManagerTy RequestManager, void *HstPtr,
                         int64_t Size) {
  RequestManager.send(&HstPtr, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy dataNotifyUnmapped(MPIRequestManagerTy RequestManager, void *HstPtr) {
  RequestManager.send(&HstPtr, sizeof(void *), MPI_BYTE);
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

EventTy exit(MPIRequestManagerTy RequestManager) {
  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

} // namespace OriginEvents

/// Event Queue implementation
EventQueue::EventQueue() : Queue(), QueueMtx(), CanPopCv() {}

size_t EventQueue::size() {
  std::lock_guard<std::mutex> Lock(QueueMtx);
  return Queue.size();
}

void EventQueue::push(EventTy &&Event) {
  {
    std::unique_lock<std::mutex> Lock(QueueMtx);
    Queue.emplace(Event);
  }

  // Notifies a thread possibly blocked by an empty queue.
  CanPopCv.notify_one();
}

EventTy EventQueue::pop(std::stop_token &Stop) {
  std::unique_lock<std::mutex> Lock(QueueMtx);

  // Waits for at least one item to be pushed.
  const bool HasNewEvent =
      CanPopCv.wait(Lock, Stop, [&] { return !Queue.empty(); });

  if (!HasNewEvent) {
    assert(Stop.stop_requested() && "Queue was empty while running.");
    return EventTy();
  }

  EventTy TargetEvent = std::move(Queue.front());
  Queue.pop();
  return TargetEvent;
}

/// Event System implementation
EventSystemTy::EventSystemTy()
    : EventSystemState(EventSystemStateTy::CREATED),
      NumMPIComms("OMPTARGET_NUM_MPI_COMMS", 10) {}

EventSystemTy::~EventSystemTy() {
  if (EventSystemState == EventSystemStateTy::FINALIZED)
    return;

  REPORT("Destructing internal event system before deinitializing it.\n");
  deinitialize();
}

bool EventSystemTy::initialize() {
  if (EventSystemState >= EventSystemStateTy::INITIALIZED) {
    REPORT("Trying to initialize event system twice.\n");
    return false;
  }

  if (!createLocalMPIContext())
    return false;

  EventSystemState = EventSystemStateTy::INITIALIZED;

  return true;
}

bool EventSystemTy::is_initialized() {
  return EventSystemState == EventSystemStateTy::INITIALIZED;
}

bool EventSystemTy::deinitialize() {
  if (EventSystemState == EventSystemStateTy::FINALIZED) {
    REPORT("Trying to deinitialize event system twice.\n");
    return false;
  }

  if (EventSystemState == EventSystemStateTy::RUNNING) {
    REPORT("Trying to deinitialize event system while it is running.\n");
    return false;
  }

  // Only send exit events from the host side
  if (isHost() && WorldSize > 1) {
    const int NumWorkers = WorldSize - 1;
    llvm::SmallVector<EventTy> ExitEvents(NumWorkers);
    for (int WorkerRank = 0; WorkerRank < NumWorkers; WorkerRank++) {
      ExitEvents[WorkerRank] =
          createEvent(OriginEvents::exit, EventTypeTy::EXIT, WorkerRank);
      ExitEvents[WorkerRank].resume();
    }

    bool SuccessfullyExited = true;
    for (int WorkerRank = 0; WorkerRank < NumWorkers; WorkerRank++) {
      ExitEvents[WorkerRank].wait();
      SuccessfullyExited &= ExitEvents[WorkerRank].done();
      auto Error = ExitEvents[WorkerRank].getError();
      if (Error)
        REPORT("Exit event failed with msg: %s\n",
               toString(std::move(Error)).data());
    }

    if (!SuccessfullyExited) {
      REPORT("Failed to stop worker processes.\n");
      return false;
    }
  }

  if (!destroyLocalMPIContext())
    return false;

  EventSystemState = EventSystemStateTy::FINALIZED;

  return true;
}

EventTy EventSystemTy::createExchangeEvent(int SrcDevice, const void *SrcBuffer,
                                           int DstDevice, void *DstBuffer,
                                           int64_t Size,
                                           __tgt_async_info *AsyncInfo) {
  const int EventTag = createNewEventTag();
  auto &EventComm = getNewEventComm(EventTag);

  int32_t SrcRank, SrcDeviceId, DstRank, DstDeviceId;

  std::tie(SrcRank, SrcDeviceId) = mapDeviceId(SrcDevice);
  std::tie(DstRank, DstDeviceId) = mapDeviceId(DstDevice);

  int SrcEventNotificationInfo[] = {static_cast<int>(EventTypeTy::EXCHANGE_SRC),
                                    EventTag, SrcDeviceId};

  int DstEventNotificationInfo[] = {static_cast<int>(EventTypeTy::EXCHANGE_DST),
                                    EventTag, DstDeviceId};

  MPI_Request SrcRequest = MPI_REQUEST_NULL;
  MPI_Request DstRequest = MPI_REQUEST_NULL;

  int MPIError = MPI_Isend(SrcEventNotificationInfo, 3, MPI_INT, SrcRank,
                           static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                           GateThreadComm, &SrcRequest);

  MPIError &= MPI_Isend(DstEventNotificationInfo, 3, MPI_INT, DstRank,
                        static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                        GateThreadComm, &DstRequest);

  if (MPIError != MPI_SUCCESS)
    co_return createError(
        "MPI failed during exchange event notification with error %d",
        MPIError);

  MPIRequestManagerTy RequestManager(EventComm, EventTag, SrcRank, SrcDeviceId,
                                     {SrcRequest, DstRequest});

  co_return (co_await OriginEvents::exchange(std::move(RequestManager), SrcRank,
                                             SrcBuffer, DstRank, DstBuffer,
                                             Size, AsyncInfo));
}

/// Creates a new event tag of at least 'FIRST_EVENT' value.
/// Tag values smaller than 'FIRST_EVENT' are reserved for control
/// messages between the event systems of different MPI processes.
int EventSystemTy::createNewEventTag() {
  int tag = 0;

  do {
    tag = EventCounter.fetch_add(1) % MPITagMaxValue;
  } while (tag < static_cast<int>(ControlTagsTy::FIRST_EVENT));

  return tag;
}

MPI_Comm &EventSystemTy::getNewEventComm(int MPITag) {
  // Retrieve a comm using a round-robin strategy around the event's mpi tag.
  return EventCommPool[MPITag % EventCommPool.size()];
}

static const char *threadLevelToString(int ThreadLevel) {
  switch (ThreadLevel) {
  case MPI_THREAD_SINGLE:
    return "MPI_THREAD_SINGLE";
  case MPI_THREAD_SERIALIZED:
    return "MPI_THREAD_SERIALIZED";
  case MPI_THREAD_FUNNELED:
    return "MPI_THREAD_FUNNELED";
  case MPI_THREAD_MULTIPLE:
    return "MPI_THREAD_MULTIPLE";
  default:
    return "unkown";
  }
}

/// Initialize the MPI context.
bool EventSystemTy::createLocalMPIContext() {
  int MPIError = MPI_SUCCESS;
  int IsMPIInitialized = 0;
  int ThreadLevel = MPI_THREAD_SINGLE;

  MPI_Initialized(&IsMPIInitialized);

  if (IsMPIInitialized)
    MPI_Query_thread(&ThreadLevel);
  else
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &ThreadLevel);

  CHECK(ThreadLevel == MPI_THREAD_MULTIPLE,
        "MPI plugin requires a MPI implementation with %s thread level. "
        "Implementation only supports up to %s.\n",
        threadLevelToString(MPI_THREAD_MULTIPLE),
        threadLevelToString(ThreadLevel));

  if (IsMPIInitialized && ThreadLevel == MPI_THREAD_MULTIPLE) {
    MPI_Comm_rank(MPI_COMM_WORLD, &LocalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &WorldSize);
    return true;
  }

  // Create gate thread comm.
  MPIError = MPI_Comm_dup(MPI_COMM_WORLD, &GateThreadComm);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to create gate thread MPI comm with error %d\n", MPIError);

  // Create event comm pool.
  EventCommPool.resize(NumMPIComms.get(), MPI_COMM_NULL);
  for (auto &Comm : EventCommPool) {
    MPI_Comm_dup(MPI_COMM_WORLD, &Comm);
    CHECK(MPIError == MPI_SUCCESS,
          "Failed to create MPI comm pool with error %d\n", MPIError);
  }

  // Get local MPI process description.
  MPIError = MPI_Comm_rank(GateThreadComm, &LocalRank);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to acquire the local MPI rank with error %d\n", MPIError);

  MPIError = MPI_Comm_size(GateThreadComm, &WorldSize);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to acquire the world size with error %d\n", MPIError);

  // Get max value for MPI tags.
  MPI_Aint *Value = nullptr;
  int Flag = 0;
  MPIError = MPI_Comm_get_attr(GateThreadComm, MPI_TAG_UB, &Value, &Flag);
  CHECK(Flag == 1 && MPIError == MPI_SUCCESS,
        "Failed to acquire the MPI max tag value with error %d\n", MPIError);
  MPITagMaxValue = *Value;

  return true;
}

/// Destroy the MPI context.
bool EventSystemTy::destroyLocalMPIContext() {
  int MPIError = MPI_SUCCESS;

  if (GateThreadComm == MPI_COMM_NULL) {
    return true;
  }

  // Note: We don't need to assert here since application part of the program
  // was finished.
  // Free gate thread comm.
  MPIError = MPI_Comm_free(&GateThreadComm);
  CHECK(MPIError == MPI_SUCCESS,
        "Failed to destroy the gate thread MPI comm with error %d\n", MPIError);

  // Free event comm pool.
  for (auto &comm : EventCommPool) {
    MPI_Comm_free(&comm);
    CHECK(MPIError == MPI_SUCCESS,
          "Failed to destroy the event MPI comm with error %d\n", MPIError);
  }
  EventCommPool.resize(0);

  // Finalize the global MPI session.
  int IsFinalized = false;
  MPIError = MPI_Finalized(&IsFinalized);

  if (IsFinalized) {
    DP("MPI was already finalized externally.\n");
  } else {
    MPIError = MPI_Finalize();
    CHECK(MPIError == MPI_SUCCESS, "Failed to finalize MPI with error: %d\n",
          MPIError);
  }

  return true;
}

int EventSystemTy::getNumWorkers() const {
  if (isHost())
    return WorldSize - 1;
  return 0;
}

int EventSystemTy::isHost() const { return LocalRank == WorldSize - 1; };

/// Map DeviceId to the pair <RemoteRank, RemoteDeviceId>
RemoteDeviceId EventSystemTy::mapDeviceId(int32_t DeviceId) {
  int32_t NumRanks = DevicesPerRemote.size();
  for (int32_t RemoteRank = 0; RemoteRank < NumRanks; RemoteRank++) {
    if (DeviceId < DevicesPerRemote[RemoteRank])
      return {RemoteRank, DeviceId};
    DeviceId -= DevicesPerRemote[RemoteRank];
  }
  return {-1, -1};
}
