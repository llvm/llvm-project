//=== OmptTracingBuffer.cpp - Target independent OpenMP target RTL -- C++ -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of OMPT device trace record generation and flushing.
//
//===----------------------------------------------------------------------===//

#include "Shared/Debug.h"
#include "OmptTracing.h"
#include "OmptTracingBuffer.h"
#include "private.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <limits>

// When set to true, helper threads terminate their work
static bool done_tracing{false};

// Unique buffer id in creation order
static std::atomic<uint64_t> buf_id{0};

// Unique id in buffer flush order
static std::atomic<uint64_t> flush_id{0};

thread_local OmptTracingBufferMgr::BufPtr
    OmptTracingBufferMgr::ArrayOfBufPtr[MAX_NUM_DEVICES];

static uint64_t get_and_inc_buf_id() { return buf_id++; }

static uint64_t get_and_inc_flush_id() { return flush_id++; }
static uint64_t get_flush_id() { return flush_id; }

/*
 * Used by OpenMP threads for assigning space for a trace record. If
 * there is no space in the last buffer allocated by this thread, the
 * last buffer is marked full and scheduled for flushing. Otherwise,
 * space is assigned for a trace record and the new cursor returned.
 * Since the memory allocated by a thread is used by that thread alone
 * for creating trace records, a lock need not be held. In the less
 * common branch when memory is allocated, a lock needs to be acquired
 * for updating shared metadata. The common path of allocating a trace
 * record from an existing buffer proceeds without locking.
 */
void *OmptTracingBufferMgr::assignCursor(ompt_callbacks_t Type,
                                         int64_t DeviceId) {
  // The caller should handle nullptr by not tracing for this event.
  if (DeviceId < 0 || DeviceId > MAX_NUM_DEVICES - 1)
    return nullptr;

  size_t RecSize = getTRSize();

  // If the buffer fills up, it will be scheduled for flushing with the
  // following cursor.
  void *ToBeFlushedCursor = nullptr;
  BufPtr ToBeFlushedBuf = nullptr;

  // Thread local buffer pointer should be non-null once an allocation
  // has been done by this thread.
  BufPtr DeviceBuf = getDeviceSpecificBuffer(DeviceId);
  if (DeviceBuf != nullptr) {
    assert(DeviceBuf->DeviceId == DeviceId && "Unexpected device id in buffer");
    void *OldCursor = DeviceBuf->Cursor.load(std::memory_order_acquire);
    // Try to assign a trace record from the last allocated buffer
    if (RecSize <= DeviceBuf->RemainingBytes) {
      assert((char *)DeviceBuf->Start + DeviceBuf->TotalBytes -
                 DeviceBuf->RemainingBytes ==
             (char *)OldCursor + RecSize);
      DeviceBuf->RemainingBytes -= RecSize;

      // Note the trace record status must be initialized before setting
      // the cursor, ensuring that a helper thread always sees an initialized
      // trace record status.
      void *NewCursor = (char *)OldCursor + RecSize;
      initTraceRecordMetaData(NewCursor);
      DeviceBuf->Cursor.store(NewCursor, std::memory_order_release);

      DP("Thread %lu: Assigned %lu bytes at %p in existing buffer %p for "
         "device %ld\n",
         llvm::omp::target::ompt::getThreadId(), RecSize, NewCursor,
         DeviceBuf->Start, DeviceId);
      return NewCursor;
    } else {
      ToBeFlushedCursor = OldCursor;
      ToBeFlushedBuf = DeviceBuf;

      // Mark that no space is present for any more trace records.
      // The following is atomic but there is no logical order between when
      // it is set here and when it is checked by a helper thread. That works
      // because the helper thread uses this info to decide whether a buffer
      // can be scheduled for removal. In the worst case, the buffer will be
      // removed late.
      DeviceBuf->isFull.store(true, std::memory_order_release);
    }
  }
  void *NewBuffer = nullptr;
  size_t TotalBytes = 0;
  // TODO Move the buffer allocation to a helper thread
  llvm::omp::target::ompt::ompt_callback_buffer_request(DeviceId, &NewBuffer,
                                                        &TotalBytes);

  // The caller should handle nullptr by not tracing for this event.
  if (NewBuffer == nullptr || TotalBytes < RecSize)
    return nullptr;

  uint64_t NewBufId = get_and_inc_buf_id();
  auto new_buf = std::make_shared<Buffer>(
      NewBufId, DeviceId, /*Start=*/NewBuffer, TotalBytes,
      /*RemainingBytes=*/TotalBytes - RecSize,
      /*Cursor=*/NewBuffer,
      /*isFull=*/false);

  // Initialize trace record status before publishing it to helper threads.
  initTraceRecordMetaData(new_buf->Cursor.load(std::memory_order_acquire));
  setDeviceSpecificBuffer(DeviceId, new_buf);

  // Make this trace record visible to helper threads by adding to shared
  // metadata.
  std::unique_lock<std::mutex> lck(BufferMgrMutex);
  assert(Id2BufferMap.find(NewBufId) == Id2BufferMap.end());
  Id2BufferMap[NewBufId] = new_buf;
  lck.unlock();

  // Schedule the full buffer for flushing till the corresponding cursor.
  if (ToBeFlushedCursor)
    setComplete(ToBeFlushedCursor, ToBeFlushedBuf);

  DP("Thread %lu: Assigned %lu bytes at %p in new buffer with id %lu for "
     "device %ld\n",
     llvm::omp::target::ompt::getThreadId(), RecSize, NewBuffer, NewBufId,
     DeviceId);

  return NewBuffer;
}

/*
 * Called by an OpenMP thread when a buffer fills up and should be
 * flushed. This function assigns a new flush_id to the buffer, adds
 * to the flush-related metadata and wakes up a helper thread to
 * dispatch a buffer-completion callback. This function should be
 * called without holding any lock.
 * Note lock order: buf_lock -> flush_lock
 */
void OmptTracingBufferMgr::setComplete(void *cursor, BufPtr Buf) {
  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);

  // Between calling setComplete and this check, a flush-all may have
  // delivered this buffer to the tool and deleted it. So the buffer
  // may not exist.
  if (Id2BufferMap.find(Buf->Id) == Id2BufferMap.end())
    return;

  // Cannot assert that the state of the cursor is ready since a
  // different thread may be in the process of populating it. If it
  // remains in init state when the range of trace records is
  // determined for dispatching the buffer-completion callback, it
  // will not be included.
  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  uint64_t flush_id;
  auto flush_itr = FlushBufPtr2IdMap.find(Buf);
  if (flush_itr == FlushBufPtr2IdMap.end()) {
    // This buffer has not been flushed yet
    addNewFlushEntry(Buf, cursor);
  } else {
    // This buffer has been flushed before
    flush_id = flush_itr->second;
    auto flush_md_itr = Id2FlushMdMap.find(flush_id);
    assert(flush_md_itr != Id2FlushMdMap.end());
    flush_md_itr->second.FlushCursor = cursor; // update the cursor
    // Do not update the flush status since it may be under processing
    // by another thread
    DP("Updated id %lu cursor %p buf %p\n", flush_id, cursor,
       flush_md_itr->second.FlushBuf->Start);
  }
  flush_lock.unlock();
  buf_lock.unlock();

  // Wake up a helper thread to invoke the buffer-completion callback
  FlushCv.notify_one();
}

// This is the driver routine for the completion thread
void OmptTracingBufferMgr::driveCompletion() {
  while (true) {
    bool should_signal_workers = false;
    std::unique_lock<std::mutex> flush_lock(FlushMutex);
    if (done_tracing) {
      // An upper layer serializes flush_trace and stop_trace. In
      // addition, before done_tracing is set, a flush is performed as
      // part of stop_trace. So assert that no flush is in progress.
      assert(ThreadFlushTracker == 0);
      break;
    }
    FlushCv.wait(flush_lock, [this] {
      return done_tracing ||
             (!Id2FlushMdMap.empty() &&
              llvm::omp::target::ompt::TracingActive) ||
             isThisThreadFlushWaitedUpon();
    });
    if (isThisThreadFlushWaitedUpon()) {
      resetThisThreadFlush();
      if (ThreadFlushTracker == 0)
        should_signal_workers = true;
    }
    flush_lock.unlock();

    invokeCallbacks();

    if (should_signal_workers)
      ThreadFlushCv.notify_all();

    // There is a scenario where a buffer was processed but not full
    // or owned, so it was put back in waiting state. So this thread
    // would not wait but keep on looping without having any actual
    // work until new trace records are added and this thread
    // signaled. Hence, this thread yields.
    std::this_thread::yield();
  }
  bool is_last_helper = false;
  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  assert(done_tracing && "Helper thread exiting but not yet done");
  assert(isThisThreadShutdownWaitedUpon() &&
         "Helper thread exiting but not waited upon");
  resetThisThreadShutdown();
  if (ThreadShutdownTracker == 0)
    is_last_helper = true;
  flush_lock.unlock();
  if (is_last_helper)
    ThreadShutdownCv.notify_all();

  // Note that some trace records may have been written but not
  // delivered to the tool. If flush/stop APIs are not called by the
  // tool, those trace records may never be delivered to the tool and
  // the corresponding buffers not reclaimed. TODO Explore whether
  // this cleanup must be done.
}

/*
 * Called by a buffer-completion helper thread. This function examines
 * the flushed buffers in flush order and dispatches
 * callbacks. Lock holding is minimized by reserving a buffer,
 * processing it, and then unreserving it if there are more trace
 * records to flush later. If all trace records are flushed, a
 * callback is dispatched informing the tool that the buffer can be
 * deallocated. If the buffer can be deallocated, all metadata is
 * destroyed.
 * Note that this function must be called without holding any locks.
 */
void OmptTracingBufferMgr::invokeCallbacks() {
  DP("Looking for callbacks to invoke\n");
  auto max_id = std::numeric_limits<uint64_t>::max();
  auto curr_id = max_id;
  auto end_id = get_flush_id();
  DP("End id is %lu\n", end_id);
  while (true) {
    // Set the status of the flushed buffer to in-processing so that
    // another helper thread does not process it concurrently. An
    // OpenMP worker thread may, however, populate a trace record in a
    // reserved buffer concurrently.
    FlushInfo flush_info = findAndReserveFlushedBuf(curr_id);

    // no entry found, nothing to process
    if (curr_id == max_id && flush_info.FlushCursor == nullptr)
      return;

    if (flush_info.FlushCursor != nullptr) {
      // increment curr_id to get the candidate for the next iteration
      curr_id = flush_info.FlushId + 1;
    } else {
      assert(curr_id != max_id && "Cannot increment max id");
      ++curr_id;
    }

    DP("Next id will be %lu\n", curr_id);

    if (flush_info.FlushCursor == nullptr) {
      // This buffer must have been processed already
      if (curr_id < end_id)
        continue;
      else
        return; // nothing else to process
    }

    DP("Buf %p Cursor %p Id %lu will be flushed\n", flush_info.FlushBuf->Start,
       flush_info.FlushCursor, flush_info.FlushId);

    // Examine the status of the trace records and dispatch
    // buffer-completion callbacks as appropriate.
    flushBuffer(flush_info);

    // TODO optimize to set buffer-owned in the same pass above.
    // Currently, this is the only way a buffer is deallocated
    if (isBufferFull(flush_info)) {
      // All trace records have been delivered to the tool
      if (isBufferOwned(flush_info)) {
        // erase element from buffer and flush maps
        destroyFlushedBuf(flush_info);

        // dispatch callback with a null range and have the tool
        // deallocate the buffer
        dispatchBufferOwnedCallback(flush_info);
      } else {
        unreserveFlushedBuf(flush_info);
      }
    } else {
      unreserveFlushedBuf(flush_info);
    }
    if (curr_id >= end_id)
      return;
  }
}

/*
 * This function is called on a buffer that is already reserved by
 * this thread. Buffer-completion callbacks are dispatched for every
 * range of trace records that are ready.
 * This routine must be called without holding locks
 */
void OmptTracingBufferMgr::flushBuffer(FlushInfo flush_info) {
  assert(flush_info.FlushBuf && "Cannot flush an empty buffer");
  assert(flush_info.FlushCursor && "Cannot flush upto a null cursor");

  void *curr_tr = flush_info.FlushBuf->Start;
  void *last_tr = flush_info.FlushCursor;
  // Compute a range [first_cursor,last_cursor] to flush
  void *first_cursor = nullptr;
  void *last_cursor = nullptr;
  while (curr_tr <= last_tr) {
    TRStatus tr_status = getTRStatus(curr_tr);
    if (tr_status == TR_init || tr_status == TR_released) {
      if (first_cursor == nullptr) {
        // This TR won't be part of a range
        assert(last_cursor == nullptr &&
               "Begin/last cursors mutually inconsistent");
      } else {
        // End the current interval
        dispatchCallback(flush_info.FlushBuf->DeviceId,
                         flush_info.FlushBuf->Start, first_cursor, last_cursor);
        first_cursor = last_cursor = nullptr;
      }
    } else {
      assert(tr_status == TR_ready && "Unknown trace record status");
      setTRStatus(curr_tr, TR_released);
      if (first_cursor == nullptr)
        first_cursor = curr_tr;
      last_cursor = curr_tr;
    }
    curr_tr = getNextTR(curr_tr);
  }
  if (first_cursor != nullptr) {
    assert(last_cursor != nullptr);
    dispatchCallback(flush_info.FlushBuf->DeviceId, flush_info.FlushBuf->Start,
                     first_cursor, last_cursor);
  }
}

// Given a range of trace records, dispatch a buffer-completion callback
void OmptTracingBufferMgr::dispatchCallback(int64_t DeviceId, void *buffer,
                                            void *first_cursor,
                                            void *last_cursor) {
  assert(first_cursor != nullptr && last_cursor != nullptr &&
         "Callback with nullptr");
  addLastCursor(last_cursor);

  // This is best effort.
  // There is a small window when the buffer-completion callback may
  // be invoked even after tracing has been disabled.
  // Note that we don't want to hold a lock when dispatching the callback.
  if (llvm::omp::target::ompt::TracingActive) {
    DP("Dispatch callback w/ range (inclusive) to be flushed: %p -> %p\n",
       first_cursor, last_cursor);
    llvm::omp::target::ompt::ompt_callback_buffer_complete(
        DeviceId, buffer,
        /* bytes returned in this callback */
        (char *)getNextTR(last_cursor) - (char *)first_cursor,
        (ompt_buffer_cursor_t)first_cursor, false /* buffer_owned */);
  }

  removeLastCursor(last_cursor);
}

// Dispatch a buffer-completion callback with buffer_owned set so that
// the tool can deallocate the buffer
void OmptTracingBufferMgr::dispatchBufferOwnedCallback(
    const FlushInfo &flush_info) {
  // This is best effort.
  // There is a small window when the buffer-completion callback may
  // be invoked even after tracing has been disabled.
  // Note that we don't want to hold a lock when dispatching the callback.
  if (llvm::omp::target::ompt::TracingActive) {
    DP("Dispatch callback with buffer %p owned\n", flush_info.FlushBuf->Start);
    llvm::omp::target::ompt::ompt_callback_buffer_complete(
        flush_info.FlushBuf->DeviceId, flush_info.FlushBuf->Start, 0,
        (ompt_buffer_cursor_t)0, true /* buffer owned */);
  }
}

void OmptTracingBufferMgr::initTraceRecordMetaData(void *Rec) {
  setTRStatus(Rec, TR_init);
}

OmptTracingBufferMgr::BufPtr
OmptTracingBufferMgr::getDeviceSpecificBuffer(int64_t DeviceId) {
  if (DeviceId < 0 || DeviceId > MAX_NUM_DEVICES - 1) {
    REPORT("getDeviceSpecificBuffer: Device id %ld invalid or exceeds "
           "supported max: %d\n",
           DeviceId, MAX_NUM_DEVICES - 1);
    return nullptr;
  }
  return ArrayOfBufPtr[DeviceId];
}

void OmptTracingBufferMgr::setDeviceSpecificBuffer(int64_t DeviceId,
                                                   BufPtr Buf) {
  if (DeviceId < 0 || DeviceId > MAX_NUM_DEVICES - 1) {
    REPORT("setDeviceSpecificBuffer: Device id %ld invalid or exceeds "
           "supported max: %d\n",
           DeviceId, MAX_NUM_DEVICES - 1);
    return;
  }
  ArrayOfBufPtr[DeviceId] = Buf;
}

void OmptTracingBufferMgr::setTRStatus(void *Rec, TRStatus Status) {
  TraceRecord *TR = static_cast<TraceRecord *>(Rec);
  TR->TRState.store(Status, std::memory_order_release);
}

OmptTracingBufferMgr::TRStatus OmptTracingBufferMgr::getTRStatus(void *Rec) {
  return static_cast<TraceRecord *>(Rec)->TRState.load(
      std::memory_order_acquire);
}

void *OmptTracingBufferMgr::getNextTR(void *Rec) {
  size_t rec_size = getTRSize();
  // warning: no overflow check done
  return (char *)Rec + rec_size;
}

bool OmptTracingBufferMgr::isBufferFull(const FlushInfo &flush_info) {
  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
  return flush_info.FlushBuf->isFull;
}

void *OmptTracingBufferMgr::getBufferCursor(BufPtr buf) {
  return buf->Cursor.load(std::memory_order_acquire);
}

/*
 * Traverse all the trace records of a buffer and return true if all
 * of them have been released to the tool, otherwise return false
 */
bool OmptTracingBufferMgr::isBufferOwned(const FlushInfo &flush_info) {
  assert(isBufferFull(flush_info) && "Compute buffer-owned when it is full");
  void *curr_tr = flush_info.FlushBuf->Start;
  // Since the buffer is full, the cursor must be the last valid
  // TR. Note that this may be more up-to-date than the cursor in the
  // flush_info. Use the last valid TR to avoid dropping trace records
  void *last_tr = getBufferCursor(flush_info.FlushBuf);
  while (curr_tr <= last_tr) {
    if (getTRStatus(curr_tr) != TR_released)
      return false;
    curr_tr = getNextTR(curr_tr);
  }
  return true;
}

/*
 * A buffer must be reserved by a thread before it can be processed
 * and callbacks dispatched for that buffer. Reservation is done by
 * setting the status to in-processing.
 *
 * If a buffer is found in the flush metadata for the given id and it
 * is not in in-processing mode, reserve it by setting its mode to
 * in-processing and return the corresponding flush metadata. If the
 * given id is set to max, return the first waiting buffer in the
 * list of buffers to be flushed.
 */
OmptTracingBufferMgr::FlushInfo
OmptTracingBufferMgr::findAndReserveFlushedBuf(uint64_t flush_id) {
  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  MapId2Md::iterator flush_itr;
  if (flush_id == std::numeric_limits<uint64_t>::max()) {
    // Reserve the first waiting buffer and return it
    if (Id2FlushMdMap.empty())
      return FlushInfo();
    for (flush_itr = Id2FlushMdMap.begin(); flush_itr != Id2FlushMdMap.end();
         ++flush_itr) {
      // Reserve only if waiting
      if (flush_itr->second.FlushStatus == Flush_waiting)
        break;
    }
    if (flush_itr == Id2FlushMdMap.end())
      return FlushInfo();
  } else {
    flush_itr = Id2FlushMdMap.find(flush_id);
    if (flush_itr == Id2FlushMdMap.end() ||
        flush_itr->second.FlushStatus == Flush_processing)
      return FlushInfo();
  }
  assert(flush_itr->second.FlushStatus == Flush_waiting);
  flush_itr->second.FlushStatus = Flush_processing;
  FlushInfo flush_info(flush_itr->first, flush_itr->second.FlushCursor,
                       flush_itr->second.FlushBuf);
  DP("Reserved buffer: flush_id:%lu, cursor:%p, buf:%p\n", flush_itr->first,
     flush_itr->second.FlushCursor, flush_itr->second.FlushBuf->Start);
  return flush_info;
}

/*
 * Given a buffer, verify that it is in processing state and set its
 * status to waiting, removing the reservation. The same thread that
 * reserved it should be unreserving it but currently there is no such
 * check.
 */
void OmptTracingBufferMgr::unreserveFlushedBuf(const FlushInfo &flush_info) {
  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  auto itr = Id2FlushMdMap.find(flush_info.FlushId);
  assert(itr != Id2FlushMdMap.end() &&
         itr->second.FlushStatus == Flush_processing);
  itr->second.FlushStatus = Flush_waiting;
  DP("Unreserved buffer: flush_id:%lu, cursor:%p, buf:%p\n", flush_info.FlushId,
     flush_info.FlushCursor, flush_info.FlushBuf->Start);
}

/*
 * This function must be called after all of the trace records in the
 * buffer have been released to the tool. The buffer is removed from
 * all metadata maps.
 * Note lock order: buf_lock -> flush_lock
 */
void OmptTracingBufferMgr::destroyFlushedBuf(const FlushInfo &flush_info) {
  DP("Destroying buffer: flush_id:%lu, cursor:%p, buf:%p\n", flush_info.FlushId,
     flush_info.FlushCursor, flush_info.FlushBuf->Start);

  BufPtr buf = flush_info.FlushBuf;

  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
  Id2BufferMap.erase(buf->Id);

  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  auto flush_itr = Id2FlushMdMap.find(flush_info.FlushId);
  assert(flush_itr != Id2FlushMdMap.end());
  assert(flush_itr->second.FlushBuf == buf);
  Id2FlushMdMap.erase(flush_itr);
  FlushBufPtr2IdMap.erase(buf);
}

/*
 * Generate a new flush id and add the buffer to the flush metadata
 * maps. This function must be called while holding the flush lock.
 */
uint64_t OmptTracingBufferMgr::addNewFlushEntry(BufPtr buf, void *cursor) {
  assert(FlushBufPtr2IdMap.find(buf) == FlushBufPtr2IdMap.end());
  uint64_t flush_id = get_and_inc_flush_id();
  FlushBufPtr2IdMap.emplace(buf, flush_id);
  assert(Id2FlushMdMap.find(flush_id) == Id2FlushMdMap.end());
  Id2FlushMdMap.emplace(flush_id, FlushMd(cursor, buf, Flush_waiting));

  DP("Added new flush id %lu cursor %p buf %p\n", flush_id, cursor, buf->Start);

  return flush_id;
}

/*
 * Called by ompt_flush_trace and ompt_stop_trace. Traverse the
 * existing buffers in creation order and flush all the ready TRs
 */
int OmptTracingBufferMgr::flushAllBuffers(int DeviceId) {
  DP("Flushing buffers for device %d\n", DeviceId);
  // Overloading MAX_NUM_DEVICES to mean all devices.
  if (DeviceId < 0 || DeviceId > MAX_NUM_DEVICES)
    return 0; // failed to flush

  if (!areHelperThreadsAvailable())
    return 0; // failed to flush

  // If flush is called from a helper thread, just bail out
  if (amIHelperThread())
    return 0; // failed to flush

  // To avoid holding the mutex for too long, get the ids of the first
  // and the last TRs under lock, and then go through that range,
  // holding the mutex for an individual TR
  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
  if (Id2BufferMap.empty())
    return 1; // no trace records to flush
  uint64_t curr_buf_id = Id2BufferMap.begin()->first;
  uint64_t last_buf_id = Id2BufferMap.rbegin()->first;
  buf_lock.unlock();

  while (curr_buf_id <= last_buf_id) {
    std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
    // Another thread may have deleted this buffer by now
    auto buf_itr = Id2BufferMap.find(curr_buf_id);
    if (buf_itr == Id2BufferMap.end()) {
      ++curr_buf_id;
      continue;
    }
    BufPtr curr_buf = buf_itr->second;

    // If the device-id does not match, skip it. A device-id of MAX_NUM_DEVICES
    // indicates flushing for all devices.
    if (DeviceId != MAX_NUM_DEVICES && curr_buf->DeviceId != DeviceId) {
      ++curr_buf_id;
      continue;
    }

    // If this buffer is in the flush-map, skip it. It is either in
    // process by another thread or will be processed
    std::unique_lock<std::mutex> flush_lock(FlushMutex);
    auto flush_itr = FlushBufPtr2IdMap.find(curr_buf);
    if (flush_itr != FlushBufPtr2IdMap.end()) {
      ++curr_buf_id;
      continue;
    }
    // This buffer has not been flushed yet
    void *CurrBufCursor = getBufferCursor(curr_buf);
    uint64_t flush_id = addNewFlushEntry(curr_buf, CurrBufCursor);
    (void)flush_id; // Silence warning.
    DP("flushAllBuffers: Added new id %lu cursor %p buf %p\n", flush_id,
       CurrBufCursor, curr_buf->Start);

    flush_lock.unlock();
    buf_lock.unlock();

    ++curr_buf_id;
  }

  // This is best effort. It is possible that some trace records are
  // not flushed when the wait is done.
  waitForFlushCompletion();

  return 1; // success
}

void OmptTracingBufferMgr::waitForFlushCompletion() {
  {
    std::unique_lock<std::mutex> flush_lock(FlushMutex);
    // Setting the flush bit for a given helper thread indicates that the worker
    // thread is ready for the helper thread to do some work.
    for (uint32_t i = 0; i < OMPT_NUM_HELPER_THREADS; ++i)
      setThreadFlush(i);
  }

  // Wake up all helper threads to invoke buffer-completion callbacks.
  FlushCv.notify_all();

  // Now wait for all helper threads  to complete flushing.
  {
    std::unique_lock<std::mutex> flush_lock(FlushMutex);
    ThreadFlushCv.wait(flush_lock, [this] { return ThreadFlushTracker == 0; });
  }
}

void OmptTracingBufferMgr::init() {
  for (int i = 0; i < MAX_NUM_DEVICES; ++i)
    ArrayOfBufPtr[i] = nullptr;
  ThreadFlushTracker = 0;
  ThreadShutdownTracker = 0;
  done_tracing = false; // TODO make it a class member
}

void OmptTracingBufferMgr::startHelperThreads() {
  // All helper threads are stopped while holding FlushMutex. So if
  // any helper thread is present, just return. This takes care of
  // repeated calls to start-trace.
  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  if (!HelperThreadIdMap.empty()) {
    assert(!done_tracing && "Helper threads exist but tracing is done");
    return;
  }
  init();
  createHelperThreads();
}

bool OmptTracingBufferMgr::areHelperThreadsAvailable() {
  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  if (done_tracing // If another thread called stop, assume there are no threads
      || HelperThreadIdMap.empty() // Threads were never started
  ) {
    // Don't assert on HelperThreadIdMap since shutdown by another
    // thread may be in progress
    return false;
  }
  return true;
}

void OmptTracingBufferMgr::shutdownHelperThreads() {
  if (!areHelperThreadsAvailable())
    return;

  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  // If I am destroying the threads, then at least one thread must be present
  assert(!CompletionThreads.empty());
  assert(!HelperThreadIdMap.empty());
  assert(ThreadShutdownTracker == 0);

  // Set the done flag which helper threads will look at
  done_tracing = true;
  // Wait to make sure all helper threads exit
  for (uint32_t i = 0; i < OMPT_NUM_HELPER_THREADS; ++i)
    setThreadShutdown(i);
  // Signal indicating that done_tracing is set
  FlushCv.notify_all();
  ThreadShutdownCv.wait(flush_lock,
                        [this] { return ThreadShutdownTracker == 0; });

  // Now destroy all the helper threads
  destroyHelperThreads();
}

void OmptTracingBufferMgr::flushAndShutdownHelperThreads() {
  std::unique_lock<std::mutex> Lock(llvm::omp::target::ompt::TraceControlMutex);
  // Flush buffers for all devices.
  flushAllBuffers(MAX_NUM_DEVICES);
  shutdownHelperThreads();
}

void OmptTracingBufferMgr::createHelperThreads() {
  for (uint32_t i = 0; i < OMPT_NUM_HELPER_THREADS; ++i) {
    CompletionThreads.emplace_back(
        std::thread(&OmptTracingBufferMgr::driveCompletion, this));
    HelperThreadIdMap[CompletionThreads.back().get_id()] = i;
  }
}

void OmptTracingBufferMgr::destroyHelperThreads() {
  for (auto &thd : CompletionThreads)
    thd.join();
  CompletionThreads.clear();
  HelperThreadIdMap.clear();
}

OmptTracingBufferMgr::OmptTracingBufferMgr() {
  // no need to hold locks for init() since object is getting constructed here
  init();
}
