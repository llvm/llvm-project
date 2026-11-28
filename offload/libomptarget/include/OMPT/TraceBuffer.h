//===- TraceBuffer.h - Generic device trace record buffering --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A buffer of fixed-size, type-erased slots, carved out of provider-supplied
// buffers by producer threads and delivered to a consumer once ready.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OMPT_TRACEBUFFER_H
#define OMPTARGET_OMPT_TRACEBUFFER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

/// Settings of a TraceBufferCore, fixed at construction.
struct TraceBufferConfig {
  /// Queue a buffer for processing as soon as it is full.
  bool FlushOnBufferFull = true;
  /// flushAndStopHelper flushes all devices; checked at destruction.
  bool FlushOnShutdown = true;
  /// Valid device ids are [0, MaxDevices).
  int64_t MaxDevices = 1024;
  /// Longest helper wait before re-scanning queued buffers while enabled.
  std::chrono::milliseconds HelperPollInterval{10};
};

/// Supplies the buffers records are written into. allocateBuffer may run
/// concurrently on producer threads; releaseBuffer never runs concurrently with
/// itself or RecordConsumer::consumeRecords.
class BufferProvider {
public:
  /// A buffer handed to the core.
  struct BufferTy {
    void *Ptr = nullptr;
    size_t Bytes = 0;
  };

  virtual ~BufferProvider() = default;

  /// Returns a malloc-aligned buffer; null or under one slot is not traced.
  virtual BufferTy allocateBuffer(int64_t DeviceId) = 0;

  /// Takes back Buffer once all its slots were consumed; it may be freed.
  virtual void releaseBuffer(int64_t DeviceId, void *Buffer) = 0;
};

/// Receives published records. Calls never run concurrently with each other or
/// with BufferProvider::releaseBuffer.
class RecordConsumer {
public:
  virtual ~RecordConsumer() = default;

  /// Consumes the contiguous run of records [First, First + Bytes) in Buffer.
  virtual void consumeRecords(int64_t DeviceId, void *Buffer, void *First,
                              size_t Bytes) = 0;
};

/// Hands out fixed-size slots carved from provider buffers and passes
/// published records to the consumer, on the helper thread or inline in flush.
/// Callers publish each slot at most once; an unpublished slot keeps its buffer
/// from being released. Provider and consumer outlive the core. flush/flushAll
/// never overlap startHelper/stopHelper/flushAndStopHelper. Before destruction
/// the helper is stopped and, with FlushOnShutdown, all records are flushed.
/// consumeRecords and releaseBuffer do not start, stop, process or destroy it.
class TraceBufferCore {
public:
  /// Slot lifecycle: allocated, published, handed to the consumer.
  enum class SlotState : uint8_t { Init, Ready, Released };
  using StatusTy = std::atomic<SlotState>;

  /// Byte layout of a slot.
  struct LayoutTy {
    /// Slot size; a multiple of alignof(StatusTy).
    size_t Stride;
    /// Offset of the slot's StatusTy, aligned for it.
    size_t StatusOffset;
  };

  /// Creates a core with no helper running and helper processing disabled.
  TraceBufferCore(LayoutTy Layout, BufferProvider &Provider,
                  RecordConsumer &Consumer, TraceBufferConfig Config = {});
  TraceBufferCore(const TraceBufferCore &) = delete;
  TraceBufferCore &operator=(const TraceBufferCore &) = delete;
  /// Joins a running helper; makes no provider or consumer calls.
  ~TraceBufferCore();

  /// Returns a slot for DeviceId, or null if out of range or declined.
  void *allocSlot(int64_t DeviceId);
  /// Marks Slot ready for the consumer; any thread may publish.
  void publishSlot(void *Slot);

  /// Lets the helper process queued buffers without a flush; off by default.
  void setEnabled(bool Enabled);
  /// Consumes DeviceId's published records; false if out of range or nested.
  bool flush(int64_t DeviceId);
  /// Consumes all devices' published records; false if nested in a pass.
  bool flushAll();

  /// Starts the helper thread; no-op if it is running.
  void startHelper();
  /// Stops the helper after one final pass; no-op if it is not running.
  void stopHelper();
  /// Runs flushAll (FlushOnShutdown) or one helper pass, then stopHelper.
  void flushAndStopHelper();

  /// Runs one processing pass here; true if anything was consumed or released.
  bool processPending();

  /// Slot size in bytes.
  size_t getStride() const { return Layout.Stride; }

private:
  struct BufferInfoTy;
  using BufPtr = std::shared_ptr<BufferInfoTy>;

  /// Whether a queued buffer is reserved by a processing pass.
  enum class BufferFlushStatus : uint8_t { Waiting, Processing };

  /// A reserved flush queue entry; a null FlushCursor means none was found.
  struct FlushInfo {
    uint64_t FlushId = 0;
    void *FlushCursor = nullptr;
    BufPtr FlushBuf;
  };

  /// Flush queue entry: last slot to scan, buffer, reservation state.
  struct FlushMd {
    void *FlushCursor;
    BufPtr FlushBuf;
    BufferFlushStatus FlushStatus;
  };

  /// Status word of Slot.
  StatusTy &getStatus(void *Slot) const {
    return *reinterpret_cast<StatusTy *>(static_cast<char *>(Slot) +
                                         Layout.StatusOffset);
  }
  /// Slot after Slot in the same buffer.
  void *nextSlot(void *Slot) const {
    return static_cast<char *>(Slot) + Layout.Stride;
  }
  /// Loads Slot's state with acquire ordering.
  SlotState getSlotState(void *Slot) const;
  /// Stores Slot's state with release ordering.
  void setSlotState(void *Slot, SlotState State);
  /// Constructs Slot's status as Init.
  void initSlotStatus(void *Slot);
  /// Whether Slot is a whole, stride-aligned slot inside Buf.
  bool isSlotOf(const BufferInfoTy &Buf, const void *Slot) const;

  /// The calling thread's current buffers of this core, indexed by device;
  /// created empty on first use.
  std::vector<BufPtr> &getThreadBuffers();

  /// Queues a full, still registered buffer up to its final cursor.
  void triggerFlushOnBufferFull(void *Cursor, BufPtr Buf);
  /// Queues each registered, unqueued buffer of DeviceId or of all devices.
  void queuePendingBuffers(int64_t DeviceId, bool AllDevices);
  /// Queues buffers, then processes inline or waits for a helper pass.
  bool doFlush(int64_t DeviceId, bool AllDevices);
  /// Requests a helper pass and waits for it; Lock holds FlushMutex.
  void bumpAndWaitForCompletion(std::unique_lock<std::mutex> &Lock);
  /// Processes queued buffers in flush-id order; true if any work was done.
  bool invokeCallbacks();
  /// Passes each ready run up to Info's cursor on; true if any was passed.
  bool flushBuffer(FlushInfo Info);
  /// Passes the slots [First, Last] of Buffer to the consumer.
  void dispatchCallback(int64_t DeviceId, void *Buffer, void *First,
                        void *Last);
  /// Returns Info's buffer to the provider.
  void dispatchBufferOwnedCallback(const FlushInfo &Info);
  /// Reserves entry Id, or the first waiting one for the maximum id.
  FlushInfo findAndReserveFlushedBuf(uint64_t Id);
  /// Returns a reserved entry to waiting.
  void unreserveFlushedBuf(const FlushInfo &Info);
  /// Unregisters a retired buffer and removes its queue entry.
  void destroyFlushedBuf(const FlushInfo &Info);
  /// Queues Buf up to Cursor and returns its flush id; caller holds the locks.
  uint64_t addNewFlushEntry(BufPtr Buf, void *Cursor);
  /// Whether Info's buffer has no room left.
  bool isBufferFull(const FlushInfo &Info);
  /// Whether every slot of a full buffer was released to the consumer.
  bool isBufferOwned(const FlushInfo &Info);
  /// Last slot handed out from Buf.
  void *getBufferCursor(const BufPtr &Buf) const;
  /// Helper loop: runs passes on flush requests and, if enabled, queued work.
  void driveCompletion();
  /// Joins the helper; with FinalPass it runs one more pass before exiting.
  void stopHelperImpl(bool FinalPass);

  LayoutTy Layout;
  BufferProvider &Provider;
  RecordConsumer &Consumer;
  TraceBufferConfig Config;
  /// Unique per core; keys the thread-local current buffers.
  uint64_t InstanceId;

  /// Guards Id2BufferMap.
  std::mutex BufferMgrMutex;
  /// Registered buffers by buffer id.
  std::map<uint64_t, BufPtr> Id2BufferMap;

  /// Guards the flush queue, the generations, Done and HelperThread.
  std::mutex FlushMutex;
  /// Flush queue by flush id.
  std::map<uint64_t, FlushMd> Id2FlushMdMap;
  /// Flush id of each queued buffer.
  std::unordered_map<BufPtr, uint64_t> FlushBufPtr2IdMap;
  /// Wakes the helper.
  std::condition_variable FlushCv;
  /// Wakes flushers waiting for CompletedGen.
  std::condition_variable ThreadFlushCv;
  /// Helper passes requested by flushers.
  uint64_t RequestedGen = 0;
  /// Latest requested pass the helper finished.
  uint64_t CompletedGen = 0;
  /// Bumped per new queue entry; wakes a helper in its timed wait.
  uint64_t QueuedGen = 0;
  /// Tells the helper to exit.
  bool Done = false;
  /// Whether the exiting helper runs one more pass.
  bool FinalPassOnStop = true;

  /// Serialises processing passes, hence consumer and release calls.
  std::mutex ProcessMutex;

  /// Next buffer id.
  std::atomic<uint64_t> BufId{0};
  /// Next flush id.
  std::atomic<uint64_t> FlushId{0};
  /// Set by setEnabled.
  std::atomic<bool> Enabled{false};

  /// Lifecycle calls in progress; checked in assertion builds only.
  std::atomic<int> LifecycleCalls{0};
  /// Flushes in progress; checked in assertion builds only.
  std::atomic<int> FlushCalls{0};

  std::thread HelperThread;
};

/// Typed front end to a TraceBufferCore it owns. The record sits at slot
/// offset 0, so a consumer walks a run of records with next().
template <typename RecordT> class TraceBufferManager {
  static_assert(std::is_trivially_copyable_v<RecordT>);
  static_assert(std::is_standard_layout_v<RecordT>);

  /// Slot layout: the record, then its status.
  struct SlotTy {
    RecordT Record;
    TraceBufferCore::StatusTy Status;
  };
  static_assert(offsetof(SlotTy, Record) == 0);
  static_assert(alignof(SlotTy) <= alignof(std::max_align_t));

  /// Core layout for SlotTy.
  static constexpr TraceBufferCore::LayoutTy getLayout() {
    return {sizeof(SlotTy), offsetof(SlotTy, Status)};
  }

public:
  /// Creates the core for RecordT slots.
  TraceBufferManager(BufferProvider &Provider, RecordConsumer &Consumer,
                     TraceBufferConfig Config = {})
      : Core(getLayout(), Provider, Consumer, Config) {}

  /// Returns a record to fill for DeviceId, or null if not traced.
  RecordT *alloc(int64_t DeviceId) {
    return static_cast<RecordT *>(Core.allocSlot(DeviceId));
  }
  /// Publishes a filled record; see TraceBufferCore::publishSlot.
  void publish(RecordT *Record) { Core.publishSlot(Record); }

  /// Distance in bytes between consecutive records in a buffer.
  static constexpr size_t getStride() { return sizeof(SlotTy); }
  /// Record after Record in the same buffer.
  static RecordT *next(RecordT *Record) {
    return reinterpret_cast<RecordT *>(reinterpret_cast<char *>(Record) +
                                       getStride());
  }

  /// The owned core, for flushing and helper control.
  TraceBufferCore &getCore() { return Core; }

private:
  TraceBufferCore Core;
};

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OMPTARGET_OMPT_TRACEBUFFER_H
