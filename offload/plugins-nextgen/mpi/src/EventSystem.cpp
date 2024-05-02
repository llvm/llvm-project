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

#include <ffi.h>
#include <mpi.h>
#include <unistd.h>

#include "Shared/Debug.h"
#include "Shared/EnvironmentVar.h"
#include "Shared/Utils.h"
#include "omptarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include "llvm/Support/DynamicLibrary.h"

using llvm::sys::DynamicLibrary;

#define CHECK(expr, msg, ...)                                                  \
  if (!(expr)) {                                                               \
    REPORT(msg, ##__VA_ARGS__);                                                \
    return false;                                                              \
  }

/// Customizable parameters of the event system
///
/// Number of execute event handlers to spawn.
static IntEnvar NumExecEventHandlers("OMPTARGET_NUM_EXEC_EVENT_HANDLERS", 1);
/// Number of data event handlers to spawn.
static IntEnvar NumDataEventHandlers("OMPTARGET_NUM_DATA_EVENT_HANDLERS", 1);
/// Polling rate period (us) used by event handlers.
static IntEnvar EventPollingRate("OMPTARGET_EVENT_POLLING_RATE", 1);
/// Number of communicators to be spawned and distributed for the events.
/// Allows for parallel use of network resources.
static Int64Envar NumMPIComms("OMPTARGET_NUM_MPI_COMMS", 10);
/// Maximum buffer Size to use during data transfer.
static Int64Envar MPIFragmentSize("OMPTARGET_MPI_FRAGMENT_SIZE", 100e6);

/// Helper function to transform event type to string
const char *toString(EventTypeTy Type) {
  using enum EventTypeTy;

  switch (Type) {
  case ALLOC:
    return "Alloc";
  case DELETE:
    return "Delete";
  case RETRIEVE:
    return "Retrieve";
  case SUBMIT:
    return "Submit";
  case EXCHANGE:
    return "Exchange";
  case EXCHANGE_SRC:
    return "exchangeSrc";
  case EXCHANGE_DST:
    return "ExchangeDst";
  case EXECUTE:
    return "Execute";
  case SYNC:
    return "Sync";
  case LOAD_BINARY:
    return "LoadBinary";
  case EXIT:
    return "Exit";
  }

  assert(false && "Every enum value must be checked on the switch above.");
  return nullptr;
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
    resume();

    std::this_thread::sleep_for(
        std::chrono::microseconds(EventPollingRate.get()));
  }
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
void MPIRequestManagerTy::sendInBatchs(void *Buffer, int Size) {
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
void MPIRequestManagerTy::receiveInBatchs(void *Buffer, int Size) {
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
  void *HstPrt = nullptr;
  int MPIError = MPI_Alloc_mem(Size, MPI_INFO_NULL, &HstPrt);
  if (MPIError != MPI_SUCCESS)
    return nullptr;
  return HstPrt;
}

int memFreeHost(void *HstPtr) {
  int MPIError = MPI_Free_mem(HstPtr);
  if (MPIError != MPI_SUCCESS)
    return OFFLOAD_FAIL;
  return OFFLOAD_SUCCESS;
}

/// Device Image Storage. This class is used to store Device Image data
/// in the remote device process.
struct DeviceImage : __tgt_device_image {
  llvm::SmallVector<unsigned char, 1> ImageBuffer;
  llvm::SmallVector<__tgt_offload_entry, 16> Entries;
  llvm::SmallVector<char> FlattenedEntryNames;

  DeviceImage() {
    ImageStart = nullptr;
    ImageEnd = nullptr;
    EntriesBegin = nullptr;
    EntriesEnd = nullptr;
  }

  DeviceImage(size_t ImageSize, size_t EntryCount)
      : ImageBuffer(ImageSize + alignof(void *)), Entries(EntryCount) {
    // Align the image buffer to alignof(void *).
    ImageStart = ImageBuffer.begin();
    std::align(alignof(void *), ImageSize, ImageStart, ImageSize);
    ImageEnd = (void *)((size_t)ImageStart + ImageSize);
  }

  void setImageEntries(llvm::SmallVector<size_t> EntryNameSizes) {
    // Adjust the entry names to use the flattened name buffer.
    size_t EntryCount = Entries.size();
    size_t TotalNameSize = 0;
    for (size_t I = 0; I < EntryCount; I++) {
      TotalNameSize += EntryNameSizes[I];
    }
    FlattenedEntryNames.resize(TotalNameSize);

    for (size_t I = EntryCount; I > 0; I--) {
      TotalNameSize -= EntryNameSizes[I - 1];
      Entries[I - 1].name = &FlattenedEntryNames[TotalNameSize];
    }

    // Set the entries pointers.
    EntriesBegin = Entries.begin();
    EntriesEnd = Entries.end();
  }

  /// Get the image size.
  size_t getSize() const {
    return llvm::omp::target::getPtrDiff(ImageEnd, ImageStart);
  }

  /// Getter and setter for the dynamic library.
  DynamicLibrary &getDynamicLibrary() { return DynLib; }
  void setDynamicLibrary(const DynamicLibrary &Lib) { DynLib = Lib; }

private:
  DynamicLibrary DynLib;
};

/// Event implementations on Host side.
namespace OriginEvents {

EventTy allocateBuffer(MPIRequestManagerTy RequestManager, int64_t Size,
                       void **Buffer) {
  RequestManager.send(&Size, 1, MPI_INT64_T);

  // Event completion notification
  RequestManager.receive(Buffer, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy deleteBuffer(MPIRequestManagerTy RequestManager, void *Buffer) {
  RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy submit(MPIRequestManagerTy RequestManager, EventDataHandleTy DataHandle,
               void *DstBuffer, int64_t Size) {
  RequestManager.send(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);

  RequestManager.sendInBatchs(DataHandle.get(), Size);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy retrieve(MPIRequestManagerTy RequestManager, void *OrgBuffer,
                 const void *DstBuffer, int64_t Size) {
  RequestManager.send(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.receiveInBatchs(OrgBuffer, Size);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy exchange(MPIRequestManagerTy RequestManager, int SrcDevice,
                 const void *OrgBuffer, int DstDevice, void *DstBuffer,
                 int64_t Size) {
  // Send data to SrcDevice
  RequestManager.send(&OrgBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&DstDevice, 1, MPI_INT);

  // Send data to DstDevice
  RequestManager.OtherRank = DstDevice;
  RequestManager.send(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.send(&Size, 1, MPI_INT64_T);
  RequestManager.send(&SrcDevice, 1, MPI_INT);

  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  RequestManager.OtherRank = SrcDevice;
  RequestManager.receive(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy execute(MPIRequestManagerTy RequestManager, EventDataHandleTy Args,
                uint32_t NumArgs, void *Func) {
  RequestManager.send(&NumArgs, 1, MPI_UINT32_T);
  RequestManager.send(Args.get(), NumArgs * sizeof(void *), MPI_BYTE);
  RequestManager.send(&Func, sizeof(void *), MPI_BYTE);

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
                   llvm::SmallVector<void *> *DeviceImageAddrs) {
  auto &[ImageStart, ImageEnd, EntriesBegin, EntriesEnd] = *Image;

  // Send the target table sizes.
  size_t ImageSize = (size_t)ImageEnd - (size_t)ImageStart;
  size_t EntryCount = EntriesEnd - EntriesBegin;
  llvm::SmallVector<size_t> EntryNameSizes(EntryCount);

  for (size_t I = 0; I < EntryCount; I++) {
    // Note: +1 for the terminator.
    EntryNameSizes[I] = std::strlen(EntriesBegin[I].name) + 1;
  }

  RequestManager.send(&ImageSize, 1, MPI_UINT64_T);
  RequestManager.send(&EntryCount, 1, MPI_UINT64_T);
  RequestManager.send(EntryNameSizes.begin(), EntryCount, MPI_UINT64_T);

  // Send the image bytes and the table entries.
  RequestManager.send(ImageStart, ImageSize, MPI_BYTE);

  for (size_t I = 0; I < EntryCount; I++) {
    RequestManager.send(&EntriesBegin[I].addr, 1, MPI_UINT64_T);
    RequestManager.send(EntriesBegin[I].name, EntryNameSizes[I], MPI_CHAR);
    RequestManager.send(&EntriesBegin[I].size, 1, MPI_UINT64_T);
    RequestManager.send(&EntriesBegin[I].flags, 1, MPI_INT32_T);
    RequestManager.send(&EntriesBegin[I].data, 1, MPI_INT32_T);
    RequestManager.receive(&((*DeviceImageAddrs)[I]), 1, MPI_UINT64_T);
  }

  co_return (co_await RequestManager);
}

EventTy exit(MPIRequestManagerTy RequestManager) {
  // Event completion notification
  RequestManager.receive(nullptr, 0, MPI_BYTE);
  co_return (co_await RequestManager);
}

} // namespace OriginEvents

/// Event Implementations on Device side.
namespace DestinationEvents {

EventTy allocateBuffer(MPIRequestManagerTy RequestManager) {
  int64_t Size = 0;
  RequestManager.receive(&Size, 1, MPI_INT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  void *Buffer = malloc(Size);
  RequestManager.send(&Buffer, sizeof(void *), MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy deleteBuffer(MPIRequestManagerTy RequestManager) {
  void *Buffer = nullptr;
  RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  free(Buffer);

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy submit(MPIRequestManagerTy RequestManager) {
  void *Buffer = nullptr;
  int64_t Size = 0;
  RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);
  RequestManager.receive(&Size, 1, MPI_INT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.receiveInBatchs(Buffer, Size);

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy retrieve(MPIRequestManagerTy RequestManager) {
  void *Buffer = nullptr;
  int64_t Size = 0;
  RequestManager.receive(&Buffer, sizeof(void *), MPI_BYTE);
  RequestManager.receive(&Size, 1, MPI_INT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  RequestManager.sendInBatchs(Buffer, Size);

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy exchangeSrc(MPIRequestManagerTy RequestManager) {
  void *SrcBuffer;
  int64_t Size;
  int DstDevice;
  // Save head node rank
  int HeadNodeRank = RequestManager.OtherRank;

  RequestManager.receive(&SrcBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.receive(&Size, 1, MPI_INT64_T);
  RequestManager.receive(&DstDevice, 1, MPI_INT);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Set the Destination Rank in RequestManager
  RequestManager.OtherRank = DstDevice;

  // Send buffer to target device
  RequestManager.sendInBatchs(SrcBuffer, Size);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Set the HeadNode Rank to send the final notificatin
  RequestManager.OtherRank = HeadNodeRank;

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy exchangeDst(MPIRequestManagerTy RequestManager) {
  void *DstBuffer;
  int64_t Size;
  int SrcDevice;
  // Save head node rank
  int HeadNodeRank = RequestManager.OtherRank;

  RequestManager.receive(&DstBuffer, sizeof(void *), MPI_BYTE);
  RequestManager.receive(&Size, 1, MPI_INT64_T);
  RequestManager.receive(&SrcDevice, 1, MPI_INT);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Set the Source Rank in RequestManager
  RequestManager.OtherRank = SrcDevice;

  // Receive buffer from the Source device
  RequestManager.receiveInBatchs(DstBuffer, Size);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Set the HeadNode Rank to send the final notificatin
  RequestManager.OtherRank = HeadNodeRank;

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy execute(MPIRequestManagerTy RequestManager,
                __tgt_device_image &DeviceImage) {

  uint32_t NumArgs = 0;
  RequestManager.receive(&NumArgs, 1, MPI_UINT32_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  llvm::SmallVector<void *> Args(NumArgs);
  llvm::SmallVector<void *> ArgPtrs(NumArgs);

  RequestManager.receive(Args.data(), NumArgs * sizeof(uintptr_t), MPI_BYTE);
  void (*TargetFunc)(void) = nullptr;
  RequestManager.receive(&TargetFunc, sizeof(uintptr_t), MPI_BYTE);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Get Args references
  for (unsigned I = 0; I < NumArgs; I++) {
    ArgPtrs[I] = &Args[I];
  }

  ffi_cif Cif{};
  llvm::SmallVector<ffi_type *> ArgsTypes(NumArgs, &ffi_type_pointer);
  ffi_status Status = ffi_prep_cif(&Cif, FFI_DEFAULT_ABI, NumArgs,
                                   &ffi_type_void, &ArgsTypes[0]);

  if (Status != FFI_OK) {
    co_return createError("Error in ffi_prep_cif: %d", Status);
  }

  long Return;
  ffi_call(&Cif, TargetFunc, &Return, &ArgPtrs[0]);

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

EventTy loadBinary(MPIRequestManagerTy RequestManager, DeviceImage &Image) {
  // Receive the target table sizes.
  size_t ImageSize = 0;
  size_t EntryCount = 0;
  RequestManager.receive(&ImageSize, 1, MPI_UINT64_T);
  RequestManager.receive(&EntryCount, 1, MPI_UINT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  llvm::SmallVector<size_t> EntryNameSizes(EntryCount);

  RequestManager.receive(EntryNameSizes.begin(), EntryCount, MPI_UINT64_T);

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // Create the device name with the appropriate sizes and receive its content.
  Image = DeviceImage(ImageSize, EntryCount);
  Image.setImageEntries(EntryNameSizes);

  // Received the image bytes and the table entries.
  RequestManager.receive(Image.ImageStart, ImageSize, MPI_BYTE);

  for (size_t I = 0; I < EntryCount; I++) {
    RequestManager.receive(&Image.Entries[I].addr, 1, MPI_UINT64_T);
    RequestManager.receive(Image.Entries[I].name, EntryNameSizes[I], MPI_CHAR);
    RequestManager.receive(&Image.Entries[I].size, 1, MPI_UINT64_T);
    RequestManager.receive(&Image.Entries[I].flags, 1, MPI_INT32_T);
    RequestManager.receive(&Image.Entries[I].data, 1, MPI_INT32_T);
  }

  if (auto Error = co_await RequestManager; Error)
    co_return Error;

  // The code below is for CPU plugin only
  // Create a temporary file.
  char TmpFileName[] = "/tmp/tmpfile_XXXXXX";
  int TmpFileFd = mkstemp(TmpFileName);
  if (TmpFileFd == -1)
    co_return createError("Failed to create tmpfile for loading target image");

  // Open the temporary file.
  FILE *TmpFile = fdopen(TmpFileFd, "wb");
  if (!TmpFile)
    co_return createError("Failed to open tmpfile %s for loading target image",
                          TmpFileName);

  // Write the image into the temporary file.
  size_t Written = fwrite(Image.ImageStart, Image.getSize(), 1, TmpFile);
  if (Written != 1)
    co_return createError("Failed to write target image to tmpfile %s",
                          TmpFileName);

  // Close the temporary file.
  int Ret = fclose(TmpFile);
  if (Ret)
    co_return createError("Failed to close tmpfile %s with the target image",
                          TmpFileName);

  // Load the temporary file as a dynamic library.
  std::string ErrMsg;
  DynamicLibrary DynLib =
      DynamicLibrary::getPermanentLibrary(TmpFileName, &ErrMsg);

  // Check if the loaded library is valid.
  if (!DynLib.isValid())
    co_return createError("Failed to load target image: %s", ErrMsg.c_str());

  // Save a reference of the image's dynamic library.
  Image.setDynamicLibrary(DynLib);

  // Delete TmpFile
  unlink(TmpFileName);

  for (size_t I = 0; I < EntryCount; I++) {
    Image.Entries[I].addr = DynLib.getAddressOfSymbol(Image.Entries[I].name);
    RequestManager.send(&Image.Entries[I].addr, 1, MPI_UINT64_T);
  }

  co_return (co_await RequestManager);
}

EventTy exit(MPIRequestManagerTy RequestManager,
             std::atomic<EventSystemStateTy> &EventSystemState) {
  EventSystemStateTy OldState =
      EventSystemState.exchange(EventSystemStateTy::EXITED);
  assert(OldState != EventSystemStateTy::EXITED &&
         "Exit event received multiple times");

  // Event completion notification
  RequestManager.send(nullptr, 0, MPI_BYTE);

  co_return (co_await RequestManager);
}

} // namespace DestinationEvents

/// Event Queue implementation
EventQueue::EventQueue() : Queue(), QueueMtx(), CanPopCv() {}

size_t EventQueue::size() {
  std::lock_guard<std::mutex> lock(QueueMtx);
  return Queue.size();
}

void EventQueue::push(EventTy &&Event) {
  {
    std::unique_lock<std::mutex> lock(QueueMtx);
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
    : EventSystemState(EventSystemStateTy::CREATED) {}

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
      ExitEvents[WorkerRank] = createEvent(OriginEvents::exit, WorkerRank);
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
                                           int64_t Size) {
  const int EventTag = createNewEventTag();
  auto &EventComm = getNewEventComm(EventTag);

  int SrcEventNotificationInfo[] = {static_cast<int>(EventTypeTy::EXCHANGE_SRC),
                                    EventTag};

  int DstEventNotificationInfo[] = {static_cast<int>(EventTypeTy::EXCHANGE_DST),
                                    EventTag};

  MPI_Request SrcRequest = MPI_REQUEST_NULL;
  MPI_Request DstRequest = MPI_REQUEST_NULL;

  int MPIError = MPI_Isend(SrcEventNotificationInfo, 2, MPI_INT, SrcDevice,
                           static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                           GateThreadComm, &SrcRequest);

  MPIError &= MPI_Isend(DstEventNotificationInfo, 2, MPI_INT, DstDevice,
                        static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                        GateThreadComm, &DstRequest);

  if (MPIError != MPI_SUCCESS)
    co_return createError(
        "MPI failed during exchange event notification with error %d",
        MPIError);

  MPIRequestManagerTy RequestManager(EventComm, EventTag, SrcDevice,
                                     {SrcRequest, DstRequest});

  co_return (co_await OriginEvents::exchange(std::move(RequestManager),
                                             SrcDevice, SrcBuffer, DstDevice,
                                             DstBuffer, Size));
}

void EventSystemTy::runEventHandler(std::stop_token Stop, EventQueue &Queue) {
  while (EventSystemState == EventSystemStateTy::RUNNING || Queue.size() > 0) {
    EventTy Event = Queue.pop(Stop);

    // Re-checks the stop condition when no event was found.
    if (Event.empty()) {
      continue;
    }

    Event.resume();

    if (!Event.done()) {
      Queue.push(std::move(Event));
    }

    auto Error = Event.getError();
    if (Error)
      REPORT("Internal event failed with msg: %s\n",
             toString(std::move(Error)).data());
  }
}

void EventSystemTy::runGateThread() {
  // Device image to be used by this gate thread.
  DeviceImage Image;

  // Updates the event state and
  EventSystemState = EventSystemStateTy::RUNNING;

  // Spawns the event handlers.
  llvm::SmallVector<std::jthread> EventHandlers;
  EventHandlers.resize(NumExecEventHandlers.get() + NumDataEventHandlers.get());
  int EventHandlersSize = EventHandlers.size();
  auto HandlerFunction = std::bind_front(&EventSystemTy::runEventHandler, this);
  for (int Idx = 0; Idx < EventHandlersSize; Idx++) {
    EventHandlers[Idx] =
        std::jthread(HandlerFunction, std::ref(Idx < NumExecEventHandlers.get()
                                                   ? ExecEventQueue
                                                   : DataEventQueue));
  }

  // Executes the gate thread logic
  while (EventSystemState == EventSystemStateTy::RUNNING) {
    // Checks for new incoming event requests.
    MPI_Message EventReqMsg;
    MPI_Status EventStatus;
    int HasReceived = false;
    MPI_Improbe(MPI_ANY_SOURCE, static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                GateThreadComm, &HasReceived, &EventReqMsg, MPI_STATUS_IGNORE);

    // If none was received, wait for `EVENT_POLLING_RATE`us for the next
    // check.
    if (!HasReceived) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(EventPollingRate.get()));
      continue;
    }

    // Acquires the event information from the received request, which are:
    // - Event type
    // - Event tag
    // - Target comm
    // - Event source rank
    int EventInfo[2];
    MPI_Mrecv(EventInfo, 2, MPI_INT, &EventReqMsg, &EventStatus);
    const auto NewEventType = static_cast<EventTypeTy>(EventInfo[0]);
    MPIRequestManagerTy RequestManager(getNewEventComm(EventInfo[1]),
                                       EventInfo[1], EventStatus.MPI_SOURCE);

    // Creates a new receive event of 'event_type' type.
    using namespace DestinationEvents;
    using enum EventTypeTy;
    EventTy NewEvent;
    switch (NewEventType) {
    case ALLOC:
      NewEvent = allocateBuffer(std::move(RequestManager));
      break;
    case DELETE:
      NewEvent = deleteBuffer(std::move(RequestManager));
      break;
    case SUBMIT:
      NewEvent = submit(std::move(RequestManager));
      break;
    case RETRIEVE:
      NewEvent = retrieve(std::move(RequestManager));
      break;
    case EXCHANGE_SRC:
      NewEvent = exchangeSrc(std::move(RequestManager));
      break;
    case EXCHANGE_DST:
      NewEvent = exchangeDst(std::move(RequestManager));
      break;
    case EXECUTE:
      NewEvent = execute(std::move(RequestManager), Image);
      break;
    case EXIT:
      NewEvent = exit(std::move(RequestManager), EventSystemState);
      break;
    case LOAD_BINARY:
      NewEvent = loadBinary(std::move(RequestManager), Image);
      break;
    case SYNC:
    case EXCHANGE:
      assert(false && "Trying to create a local event on a remote node");
    }

    if (NewEventType == EXECUTE) {
      ExecEventQueue.push(std::move(NewEvent));
    } else {
      DataEventQueue.push(std::move(NewEvent));
    }
  }

  assert(EventSystemState == EventSystemStateTy::EXITED &&
         "Event State should be EXITED after receiving an Exit event");
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
