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

#ifndef _OMPTARGET_OMPCLUSTER_EVENT_SYSTEM_H_
#define _OMPTARGET_OMPCLUSTER_EVENT_SYSTEM_H_

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

#define MPICH_SKIP_MPICXX
#include <mpi.h>

#include "llvm/ADT/SmallVector.h"

#include "Shared/EnvironmentVar.h"
#include "Shared/Utils.h"

/// External forward declarations.
struct __tgt_device_image;

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
  // Memory management.
  ALLOC,  // Allocates a buffer at the remote process.
  DELETE, // Deletes a buffer at the remote process.

  // Data movement.
  SUBMIT,       // Sends a buffer data to a remote process.
  RETRIEVE,     // Receives a buffer data from a remote process.
  EXCHANGE,     // Wait data exchange between two remote processes.
  EXCHANGE_SRC, // SRC side of the exchange event.
  EXCHANGE_DST, // DST side of the exchange event.

  // Target region execution.
  EXECUTE, // Executes a target region at the remote process.

  // Local event used to wait on other events.
  SYNC,

  // Internal event system commands.
  LOAD_BINARY, // Transmits the binary descriptor to all workers
  EXIT         // Stops the event system execution at the remote process.
};

/// EventType to string conversion.
///
/// \returns the string representation of \p type.
const char *toString(EventTypeTy Type);

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
      return {std::shared_ptr<void>(HandlePtr, [](void *HandlePtr) {
        assert(HandlePtr);
        CoHandleTy::from_address(HandlePtr).destroy();
      })};
    }
  };

  /// Returns the external coroutine handle from the event.
  CoHandleTy getHandle() const {
    return CoHandleTy::from_address(HandlePtr.get());
  }

  /// Execution handling.
  /// Resume the coroutine execution up until the next suspension point.
  void resume();

  /// Blocks the caller thread until the coroutine is completed.
  void wait();

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

public:
  /// Target peer to send and receive messages
  int OtherRank;

  MPIRequestManagerTy(MPI_Comm Comm, int Tag, int OtherRank,
                      llvm::SmallVector<MPI_Request> InitialRequests =
                          {}) // TODO: Change to initializer_list
      : Comm(Comm), Tag(Tag), Requests(InitialRequests), OtherRank(OtherRank) {}

  /// This class should not be copied.
  MPIRequestManagerTy(const MPIRequestManagerTy &) = delete;
  MPIRequestManagerTy &operator=(const MPIRequestManagerTy &) = delete;

  MPIRequestManagerTy(MPIRequestManagerTy &&Other) noexcept
      : Comm(Other.Comm), Tag(Other.Tag), Requests(Other.Requests),
        OtherRank(Other.OtherRank) {
    Other.Requests = {};
  }

  MPIRequestManagerTy &operator=(MPIRequestManagerTy &&Other) = delete;

  ~MPIRequestManagerTy();

  /// Sends a buffer of given datatype items with determined size to target.
  void send(const void *Buffer, int Size, MPI_Datatype Datatype);

  /// Send a buffer with determined size to target in batchs.
  void sendInBatchs(void *Buffer, int Size);

  /// Receives a buffer of given datatype items with determined size from
  /// target.
  void receive(void *Buffer, int Size, MPI_Datatype Datatype);

  /// Receives a buffer with determined size from target in batchs.
  void receiveInBatchs(void *Buffer, int Size);

  /// Coroutine that waits on all internal pending requests.
  EventTy wait();
};

/// Data handle for host buffers in event. It keeps the host data even if the
/// original buffer is deallocated before the event happens.
using EventDataHandleTy = std::shared_ptr<void>;

/// Routines to alloc/dealloc pinned host memory.
///
/// Allocate \p Size of host memory and returns its ptr.
void *memAllocHost(int64_t Size);

/// Deallocate the host memory pointered by \p HstPrt.
int memFreeHost(void *HstPtr);

/// Coroutine events created at the origin rank of the event.
namespace OriginEvents {

EventTy allocateBuffer(MPIRequestManagerTy RequestManager, int64_t Size,
                       void **Buffer);
EventTy deleteBuffer(MPIRequestManagerTy RequestManager, void *Buffer);
EventTy submit(MPIRequestManagerTy RequestManager,
               EventDataHandleTy DataHandler, void *DstBuffer, int64_t Size);
EventTy retrieve(MPIRequestManagerTy RequestManager, void *OrgBuffer,
                 const void *DstBuffer, int64_t Size);
EventTy exchange(MPIRequestManagerTy RequestManager, int SrcDevice,
                 const void *OrgBuffer, int DstDevice, void *DstBuffer,
                 int64_t Size);
EventTy execute(MPIRequestManagerTy RequestManager, EventDataHandleTy Args,
                uint32_t NumArgs, void *Func);
EventTy sync(EventTy Event);
EventTy loadBinary(MPIRequestManagerTy RequestManager,
                   const __tgt_device_image *Image,
                   llvm::SmallVector<void *> *DeviceImageAddrs);
EventTy exit(MPIRequestManagerTy RequestManager);

/// Transform a function pointer to its representing enumerator.
template <typename FuncTy> constexpr EventTypeTy toEventType(FuncTy) {
  using enum EventTypeTy;

  if constexpr (std::is_same_v<FuncTy, decltype(&allocateBuffer)>)
    return ALLOC;
  else if constexpr (std::is_same_v<FuncTy, decltype(&deleteBuffer)>)
    return DELETE;
  else if constexpr (std::is_same_v<FuncTy, decltype(&submit)>)
    return SUBMIT;
  else if constexpr (std::is_same_v<FuncTy, decltype(&retrieve)>)
    return RETRIEVE;
  else if constexpr (std::is_same_v<FuncTy, decltype(&exchange)>)
    return EXCHANGE;
  else if constexpr (std::is_same_v<FuncTy, decltype(&execute)>)
    return EXECUTE;
  else if constexpr (std::is_same_v<FuncTy, decltype(&sync)>)
    return SYNC;
  else if constexpr (std::is_same_v<FuncTy, decltype(&exit)>)
    return EXIT;
  else if constexpr (std::is_same_v<FuncTy, decltype(&loadBinary)>)
    return LOAD_BINARY;

  assert(false && "Invalid event function");
}

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

private:
  /// Function executed by the event handler threads.
  void runEventHandler(std::stop_token Stop, EventQueue &Queue);

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
  bool deinitialize();

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
  EventTy createEvent(EventFuncTy EventFunc, int DstDeviceID, ArgsTy... Args);

  /// Create a new Exchange event.
  ///
  /// This function notifies \p SrcDevice and \p TargetDevice about the
  /// transfer and creates a host event that waits until the transfer is
  /// completed.
  EventTy createExchangeEvent(int SrcDevice, const void *SrcBuffer,
                              int DstDevice, void *DstBuffer, int64_t Size);

  /// Gate thread procedure.
  ///
  /// Caller thread will spawn the event handlers, execute the gate logic and
  /// wait until the event system receive an Exit event.
  void runGateThread();

  /// Get the number of workers available.
  ///
  /// \return the number of MPI available workers.
  int getNumWorkers() const;

  /// Check if we are at the host MPI process.
  ///
  /// \return true if the current MPI process is the host (rank WorldSize-1),
  /// false otherwise.
  int isHost() const;
};

template <class EventFuncTy, typename... ArgsTy>
  requires std::invocable<EventFuncTy, MPIRequestManagerTy, ArgsTy...>
EventTy EventSystemTy::createEvent(EventFuncTy EventFunc, int DstDeviceID,
                                   ArgsTy... Args) {
  // Create event MPI request manager.
  const int EventTag = createNewEventTag();
  auto &EventComm = getNewEventComm(EventTag);

  // Send new event notification.
  int EventNotificationInfo[] = {
      static_cast<int>(OriginEvents::toEventType(EventFunc)), EventTag};
  MPI_Request NotificationRequest = MPI_REQUEST_NULL;
  int MPIError = MPI_Isend(EventNotificationInfo, 2, MPI_INT, DstDeviceID,
                           static_cast<int>(ControlTagsTy::EVENT_REQUEST),
                           GateThreadComm, &NotificationRequest);

  if (MPIError != MPI_SUCCESS)
    co_return createError("MPI failed during event notification with error %d",
                          MPIError);

  MPIRequestManagerTy RequestManager(EventComm, EventTag, DstDeviceID,
                                     {NotificationRequest});

  co_return (co_await EventFunc(std::move(RequestManager), Args...));
}

#endif // _OMPTARGET_OMPCLUSTER_EVENT_SYSTEM_H_
