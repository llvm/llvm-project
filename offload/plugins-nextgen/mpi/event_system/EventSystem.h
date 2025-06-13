//===------- event_system.h - Concurrent MPI communication ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MPI Event System used by the MPI
// target.
//
//===----------------------------------------------------------------------===//

#ifndef _MPI_PROXY_EVENT_SYSTEM_H_
#define _MPI_PROXY_EVENT_SYSTEM_H_

#include <atomic>
#include <cassert>
#include <concepts>
#include <condition_variable>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>

#define MPICH_SKIP_MPICXX
#include <mpi.h>

#include "llvm/ADT/SmallVector.h"

#include "Shared/APITypes.h"
#include "Shared/EnvironmentVar.h"
#include "Shared/Utils.h"

/// External forward declarations.
struct __tgt_device_image;
struct ProxyDevice;

/// Template helper for generating llvm::Error instances from events.
template <typename... ArgsTy>
static llvm::Error createError(const char *ErrFmt, ArgsTy... Args) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), ErrFmt,
                                 Args...);
}

/// The event type (type of action it will performed).
///
/// Enumerates the available events. Each enum item should be accompanied by an
/// event class derived from BaseEvent. All the events are executed at a remote
/// MPI process target by the event.
enum class EventTypeTy : unsigned int {
  // Remote device management
  RETRIEVE_NUM_DEVICES, // Receives the number of devices from a remote process.
  INIT_DEVICE,          // Init Remote device
  INIT_RECORD_REPLAY,   // Initializes the record and replay mechanism.
  IS_PLUGIN_COMPATIBLE, // Check if the Image can be executed by the remote
                        // plugin.
  IS_DEVICE_COMPATIBLE, // Check if the Image can be executed by a device in the
                        // remote plugin.
  IS_DATA_EXCHANGABLE,  // Check if the plugin supports exchanging data.
  LOAD_BINARY,          // Transmits the binary descriptor to all workers
  GET_GLOBAL,           // Look up a global symbol in the given binary
  GET_FUNCTION,         // Look up a kernel function in the given binary.
  SYNCHRONIZE,          // Sync all events in the device.
  INIT_ASYNC_INFO,
  INIT_DEVICE_INFO,
  QUERY_ASYNC,
  PRINT_DEVICE_INFO,
  DATA_LOCK,
  DATA_UNLOCK,
  DATA_NOTIFY_MAPPED,
  DATA_NOTIFY_UNMAPPED,

  // Memory management.
  ALLOC,  // Allocates a buffer at the remote process.
  DELETE, // Deletes a buffer at the remote process.

  // Data movement.
  SUBMIT,         // Sends a buffer data to a remote process.
  RETRIEVE,       // Receives a buffer data from a remote process.
  LOCAL_EXCHANGE, // Data exchange between two devices in one remote process.
  EXCHANGE_SRC, // SRC side of the exchange event between two remote processes.
  EXCHANGE_DST, // DST side of the exchange event between two remote processes.

  // Target region execution.
  LAUNCH_KERNEL, // Executes a target region at the remote process.

  // Local event used to wait on other events.
  SYNC,

  // Internal event system commands.
  EXIT // Stops the event system execution at the remote process.
};

std::string EventTypeToString(EventTypeTy eventType);

/// Coroutine events
///
/// Return object for the event system coroutines. This class works as an
/// external handle for the coroutine execution, allowing anyone to: query for
/// the coroutine completion, resume the coroutine and check its state.
/// Moreover, this class allows for coroutines to be chainable, meaning a
/// coroutine function may wait on the completion of another one by using the
/// co_await operator, all through a single external handle.
struct EventTy {
  /// Internal event handle to access C++ coroutine states.
  struct promise_type;
  using CoHandleTy = std::coroutine_handle<promise_type>;
  std::shared_ptr<void> HandlePtr;

  /// Polling rate period (us) used by event handlers.
  IntEnvar EventPollingRate;

  /// EventType
  EventTypeTy EventType;

  /// Internal (and required) promise type. Allows for customization of the
  /// coroutines behavior and to store custom data inside the coroutine itself.
  struct promise_type {
    /// Coroutines are chained as a reverse linked-list. The most-recent
    /// coroutine in a chain points to the previous one and so on, until the
    /// root (and first) coroutine, which then points to the most-recent one.
    /// The root always refers to the coroutine stored in the external handle,
    /// the only handle an external user have access to.
    CoHandleTy PrevHandle;
    CoHandleTy RootHandle;

    /// Indicates if the coroutine was completed successfully. Contains the
    /// appropriate error otherwise.
    std::optional<llvm::Error> CoroutineError;

    promise_type() : CoroutineError(std::nullopt) {
      PrevHandle = RootHandle = CoHandleTy::from_promise(*this);
    }

    /// Event coroutines should always suspend upon creation and finalization.
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    /// Coroutines should return llvm::Error::success() or an appropriate error
    /// message.
    void return_value(llvm::Error &&GivenError) noexcept {
      CoroutineError = std::move(GivenError);
    }

    /// Any unhandled exception should create an externally visible error.
    void unhandled_exception() {
      assert(std::uncaught_exceptions() > 0 &&
             "Function should only be called if an uncaught exception is "
             "generated inside the coroutine");
      CoroutineError = createError("Event generated an unhandled exception");
    }

    /// Returns the external coroutine handle from the promise object.
    EventTy get_return_object() {
      void *HandlePtr = CoHandleTy::from_promise(*this).address();
      return {
          std::shared_ptr<void>(HandlePtr,
                                [](void *HandlePtr) {
                                  assert(HandlePtr);
                                  CoHandleTy::from_address(HandlePtr).destroy();
                                }),
          IntEnvar("OMPTARGET_EVENT_POLLING_RATE", 1), EventTypeTy::SYNC};
    }
  };

  /// Returns the external coroutine handle from the event.
  CoHandleTy getHandle() const {
    return CoHandleTy::from_address(HandlePtr.get());
  }

  /// Set Event Type
  void setEventType(EventTypeTy EType) { EventType = EType; }

  /// Get the Event Type
  EventTypeTy getEventType() const { return EventType; }

  /// Execution handling.
  /// Resume the coroutine execution up until the next suspension point.
  void resume();

  /// Blocks the caller thread until the coroutine is completed.
  void wait();

  /// Advance the coroutine execution up until the next suspension point
  /// and make the caller thread wait for `EVENT_POLLING_RATE`us for the next
  /// Check
  void advance();

  /// Checks if the coroutine is completed or not.
  bool done() const;

  /// Coroutine state handling.
  /// Checks if the coroutine is valid.
  bool empty() const;

  /// Get the returned error from the coroutine.
  llvm::Error getError() const;

  /// EventTy instances are also awaitables. This means one can link multiple
  /// EventTy together by calling the co_await operator on one another. For this
  /// to work, EventTy must implement the following three functions.
  ///
  /// Called on the new coroutine before suspending the current one on co_await.
  /// If returns true, the new coroutine is already completed, thus it should
  /// not be linked against the current one and the current coroutine can
  /// continue without suspending.
  bool await_ready() { return getHandle().done(); }

  /// Called on the new coroutine when the current one is suspended. It is
  /// responsible for chaining coroutines together.
  void await_suspend(CoHandleTy SuspendedHandle) {
    auto Handle = getHandle();
    auto &CurrPromise = Handle.promise();
    auto &SuspendedPromise = SuspendedHandle.promise();
    auto &RootPromise = SuspendedPromise.RootHandle.promise();

    CurrPromise.PrevHandle = SuspendedHandle;
    CurrPromise.RootHandle = SuspendedPromise.RootHandle;

    RootPromise.PrevHandle = Handle;
  }

  /// Called on the new coroutine when the current one is resumed. Used to
  /// return errors when co_awaiting on other EventTy.
  llvm::Error await_resume() {
    auto &Error = getHandle().promise().CoroutineError;

    if (Error) {
      return std::move(*Error);
    }

    return llvm::Error::success();
  }
};

/// Coroutine like manager for many non-blocking MPI calls. Allows for coroutine
/// to co_await on the registered MPI requests.
class MPIRequestManagerTy {
  /// Target specification for the MPI messages.
  const MPI_Comm Comm;
  const int Tag;
  /// Pending MPI requests.
  llvm::SmallVector<MPI_Request> Requests;
  /// Maximum buffer Size to use during data transfer.
  Int64Envar MPIFragmentSize;

public:
  /// Target peer to send and receive messages
  int OtherRank;

  /// Target device in OtherRank
  int DeviceId;

  int EventType;

  MPIRequestManagerTy(MPI_Comm Comm, int Tag, int OtherRank, int DeviceId,
                      llvm::SmallVector<MPI_Request> InitialRequests =
                          {}) // TODO: Change to initializer_list
      : Comm(Comm), Tag(Tag), Requests(InitialRequests),
        MPIFragmentSize("OMPTARGET_MPI_FRAGMENT_SIZE", 100e6),
        OtherRank(OtherRank), DeviceId(DeviceId), EventType(-1) {}

  /// This class should not be copied.
  MPIRequestManagerTy(const MPIRequestManagerTy &) = delete;
  MPIRequestManagerTy &operator=(const MPIRequestManagerTy &) = delete;

  MPIRequestManagerTy(MPIRequestManagerTy &&Other) noexcept
      : Comm(Other.Comm), Tag(Other.Tag), Requests(Other.Requests),
        MPIFragmentSize(Other.MPIFragmentSize), OtherRank(Other.OtherRank),
        DeviceId(Other.DeviceId), EventType(Other.EventType) {
    Other.Requests = {};
  }

  MPIRequestManagerTy &operator=(MPIRequestManagerTy &&Other) = delete;

  ~MPIRequestManagerTy();

  /// Sends a buffer of given datatype items with determined size to target.
  void send(const void *Buffer, int Size, MPI_Datatype Datatype);

  /// Send a buffer with determined size to target in batchs.
  void sendInBatchs(void *Buffer, int64_t Size);

  /// Receives a buffer of given datatype items with determined size from
  /// target.
  void receive(void *Buffer, int Size, MPI_Datatype Datatype);

  /// Receives a buffer with determined size from target in batchs.
  void receiveInBatchs(void *Buffer, int64_t Size);

  /// Coroutine that waits on all internal pending requests.
  EventTy wait();
};

EventTy operator co_await(MPIRequestManagerTy &RequestManager);

/// Data handle for host buffers in event. It keeps the host data even if the
/// original buffer is deallocated before the event happens.
using EventDataHandleTy = std::shared_ptr<void>;

/// Index Pair used to identify the remote device
using RemoteDeviceId = std::pair<int32_t, int32_t>;

/// Routines to alloc/dealloc pinned host memory.
///
/// Allocate \p Size of host memory and returns its ptr.
void *memAllocHost(int64_t Size);

/// Deallocate the host memory pointered by \p HstPrt.
int memFreeHost(void *HstPtr);

/// Coroutine events created at the origin rank of the event.
namespace OriginEvents {

EventTy retrieveNumDevices(MPIRequestManagerTy RequestManager,
                           int32_t *NumDevices);
EventTy isPluginCompatible(MPIRequestManagerTy RequestManager,
                           __tgt_device_image *Image, bool *QueryResult);
EventTy isDeviceCompatible(MPIRequestManagerTy RequestManager,
                           __tgt_device_image *Image, bool *QueryResult);
EventTy initDevice(MPIRequestManagerTy RequestManager, void **DevicePtr);
EventTy initRecordReplay(MPIRequestManagerTy RequestManager, int64_t MemorySize,
                         void *VAddr, bool IsRecord, bool SaveOutput,
                         uint64_t *ReqPtrArgOffset);
EventTy isDataExchangable(MPIRequestManagerTy RequestManager,
                          int32_t DstDeviceId, bool *QueryResult);
EventTy allocateBuffer(MPIRequestManagerTy RequestManager, int64_t Size,
                       int32_t Kind, void **Buffer);
EventTy deleteBuffer(MPIRequestManagerTy RequestManager, void *Buffer,
                     int32_t Kind);
EventTy submit(MPIRequestManagerTy RequestManager, void *TgtPtr, void *HstPtr,
               int64_t Size, __tgt_async_info *AsyncInfoPtr);
EventTy retrieve(MPIRequestManagerTy RequestManager, int64_t Size, void *HstPtr,
                 void *TgtPtr, __tgt_async_info *AsyncInfoPtr);
EventTy localExchange(MPIRequestManagerTy RequestManager, void *SrcPtr,
                      int DstDeviceId, void *DstPtr, int64_t Size,
                      __tgt_async_info *AsyncInfoPtr);
EventTy exchange(MPIRequestManagerTy RequestManager, int SrcRank,
                 const void *OrgBuffer, int DstRank, void *DstBuffer,
                 int64_t Size, __tgt_async_info *AsyncInfoPtr);
EventTy synchronize(MPIRequestManagerTy RequestManager,
                    __tgt_async_info *AsyncInfoPtr);
EventTy sync(EventTy Event);
EventTy loadBinary(MPIRequestManagerTy RequestManager,
                   const __tgt_device_image *Image,
                   __tgt_device_binary *Binary);
EventTy getGlobal(MPIRequestManagerTy RequestManager,
                  __tgt_device_binary Binary, uint64_t Size, const char *Name,
                  void **DevicePtr);
EventTy getFunction(MPIRequestManagerTy RequestManager,
                    __tgt_device_binary Binary, const char *Name,
                    void **KernelPtr);
EventTy launchKernel(MPIRequestManagerTy RequestManager, void *TgtEntryPtr,
                     EventDataHandleTy TgtArgs, EventDataHandleTy TgtOffsets,
                     EventDataHandleTy KernelArgsHandle,
                     __tgt_async_info *AsyncInfoPtr);
EventTy initAsyncInfo(MPIRequestManagerTy RequestManager,
                      __tgt_async_info **AsyncInfoPtr);
EventTy initDeviceInfo(MPIRequestManagerTy RequestManager,
                       __tgt_device_info *DeviceInfo);
EventTy queryAsync(MPIRequestManagerTy RequestManager,
                   __tgt_async_info *AsyncInfoPtr);
EventTy printDeviceInfo(MPIRequestManagerTy RequestManager);
EventTy dataLock(MPIRequestManagerTy RequestManager, void *Ptr, int64_t Size,
                 void **LockedPtr);
EventTy dataUnlock(MPIRequestManagerTy RequestManager, void *Ptr);
EventTy dataNotifyMapped(MPIRequestManagerTy RequestManager, void *HstPtr,
                         int64_t Size);
EventTy dataNotifyUnmapped(MPIRequestManagerTy RequestManager, void *HstPtr);
EventTy exit(MPIRequestManagerTy RequestManager);

} // namespace OriginEvents

/// Event Queue
///
/// Event queue for received events.
class EventQueue {
private:
  /// Base internal queue.
  std::queue<EventTy> Queue;
  /// Base queue sync mutex.
  std::mutex QueueMtx;

  /// Conditional variables to block popping on an empty queue.
  std::condition_variable_any CanPopCv;

public:
  /// Event Queue default constructor.
  EventQueue();

  /// Gets current queue size.
  size_t size();

  /// Push an event to the queue, resizing it when needed.
  void push(EventTy &&Event);

  /// Pops an event from the queue, waiting if the queue is empty. When stopped,
  /// returns a nullptr event.
  EventTy pop(std::stop_token &Stop);
};

/// Event System
///
/// MPI tags used in control messages.
///
/// Special tags values used to send control messages between event systems of
/// different processes. When adding new tags, please summarize the tag usage
/// with a side comment as done below.
enum class ControlTagsTy : int {
  EVENT_REQUEST = 0, // Used by event handlers to receive new event requests.
  FIRST_EVENT        // Tag used by the first event. Must always be placed last.
};

/// Event system execution state.
///
/// Describes the event system state through the program.
enum class EventSystemStateTy {
  CREATED,     // ES was created but it is not ready to send or receive new
               // events.
  INITIALIZED, // ES was initialized alongside internal MPI states. It is ready
               // to send new events, but not receive them.
  RUNNING,     // ES is running and ready to receive new events.
  EXITED,      // ES was stopped.
  FINALIZED    // ES was finalized and cannot run anything else.
};

/// The distributed event system.
class EventSystemTy {
  /// MPI definitions.
  /// The largest MPI tag allowed by its implementation.
  int32_t MPITagMaxValue = 0;

  /// Communicator used by the gate thread and base communicator for the event
  /// system.
  MPI_Comm GateThreadComm = MPI_COMM_NULL;

  /// Communicator pool distributed over the events. Many MPI implementations
  /// allow for better network hardware parallelism when unrelated MPI messages
  /// are exchanged over distinct communicators. Thus this pool will be given in
  /// a round-robin fashion to each newly created event to better utilize the
  /// hardware capabilities.
  llvm::SmallVector<MPI_Comm> EventCommPool{};

  /// Number of process used by the event system.
  int WorldSize = -1;

  /// The local rank of the current instance.
  int LocalRank = -1;

  /// Number of events created by the current instance so far. This is used to
  /// generate unique MPI tags for each event.
  std::atomic<int> EventCounter{0};

  /// Event queue between the local gate thread and the event handlers. The exec
  /// queue is responsible for only running the execution events, while the data
  /// queue executes all the other ones. This allows for long running execution
  /// events to not block any data transfers (which are all done in a
  /// non-blocking fashion).
  EventQueue ExecEventQueue{};
  EventQueue DataEventQueue{};

  /// Event System execution state.
  std::atomic<EventSystemStateTy> EventSystemState{};

  /// Number of communicators to be spawned and distributed for the events.
  /// Allows for parallel use of network resources.
  Int64Envar NumMPIComms;

private:
  /// Creates a new unique event tag for a new event.
  int createNewEventTag();

  /// Gets a comm for a new event from the comm pool.
  MPI_Comm &getNewEventComm(int MPITag);

  /// Creates a local MPI context containing a exclusive comm for the gate
  /// thread, and a comm pool to be used internally by the events. It also
  /// acquires the local MPI process description.
  bool createLocalMPIContext();

  /// Destroy the local MPI context and all of its comms.
  bool destroyLocalMPIContext();

public:
  EventSystemTy();
  ~EventSystemTy();

  bool initialize();
  bool is_initialized();
  bool deinitialize();

  template <class EventFuncTy, typename... ArgsTy>
    requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...>
  EventTy NotificationEvent(EventFuncTy EventFunc, EventTypeTy EventType,
                            int DstDeviceID, ArgsTy... Args);

  /// Creates a new event.
  ///
  /// Creates a new event of 'EventClass' type targeting the 'DestRank'. The
  /// 'args' parameters are additional arguments that may be passed to the
  /// EventClass origin constructor.
  ///
  /// /note: since this is a template function, it must be defined in
  /// this header.
  template <class EventFuncTy, typename... ArgsTy>
    requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...>
  EventTy createEvent(EventFuncTy EventFunc, EventTypeTy EventType,
                      int DstDeviceID, ArgsTy... Args);

  /// Create a new Exchange event.
  ///
  /// This function notifies \p SrcDevice and \p TargetDevice about the
  /// transfer and creates a host event that waits until the transfer is
  /// completed.
  EventTy createExchangeEvent(int SrcDevice, const void *SrcBuffer,
                              int DstDevice, void *DstBuffer, int64_t Size,
                              __tgt_async_info *AsyncInfo);

  /// Get the number of workers available.
  ///
  /// \return the number of MPI available workers.
  int getNumWorkers() const;

  /// Check if we are at the host MPI process.
  ///
  /// \return true if the current MPI process is the host (rank WorldSize-1),
  /// false otherwise.
  int isHost() const;

  RemoteDeviceId mapDeviceId(int32_t DeviceId);

  llvm::SmallVector<int> DevicesPerRemote{};

  friend struct ProxyDevice;
};

template <class EventFuncTy, typename... ArgsTy>
  requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...>
EventTy EventSystemTy::NotificationEvent(EventFuncTy EventFunc,
                                         EventTypeTy EventType, int DstDeviceID,
                                         ArgsTy... Args) {
  // Create event MPI request manager.
  const int EventTag = createNewEventTag();
  auto &EventComm = getNewEventComm(EventTag);

  int32_t RemoteRank = DstDeviceID, RemoteDeviceId = -1;

  if (EventType != EventTypeTy::IS_PLUGIN_COMPATIBLE &&
      EventType != EventTypeTy::RETRIEVE_NUM_DEVICES &&
      EventType != EventTypeTy::EXIT)
    std::tie(RemoteRank, RemoteDeviceId) = mapDeviceId(DstDeviceID);

  // Send new event notification.
  int EventNotificationInfo[] = {static_cast<int>(EventType), EventTag,
                                 RemoteDeviceId};
  MPI_Request NotificationRequest = MPI_REQUEST_NULL;
  int MPIError = MPI_Isend(EventNotificationInfo, 3, MPI_INT, RemoteRank,
                           static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                           GateThreadComm, &NotificationRequest);

  if (MPIError != MPI_SUCCESS)
    co_return createError("MPI failed during event notification with error %d",
                          MPIError);

  MPIRequestManagerTy RequestManager(EventComm, EventTag, RemoteRank,
                                     RemoteDeviceId, {NotificationRequest});

  RequestManager.EventType = EventNotificationInfo[0];

  auto Event = EventFunc(std::move(RequestManager), Args...);
  Event.setEventType(EventType);

  co_return (co_await Event);
}

template <class EventFuncTy, typename... ArgsTy>
  requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...>
EventTy EventSystemTy::createEvent(EventFuncTy EventFunc, EventTypeTy EventType,
                                   int DstDeviceID, ArgsTy... Args) {
  auto NEvent = NotificationEvent(EventFunc, EventType, DstDeviceID, Args...);
  NEvent.setEventType(EventType);
  return NEvent;
}

#endif // _MPI_PROXY_EVENT_SYSTEM_H_
