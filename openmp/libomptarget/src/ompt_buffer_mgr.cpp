//=== ompt_buffer_mgr.cpp - Target independent OpenMP target RTL -- C++ -*-===//
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

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <limits>

#include <Debug.h>

#include "private.h"

#include <ompt_buffer_mgr.h>
#include <ompt_device_callbacks.h>

extern ompt_device_callbacks_t ompt_device_callbacks;

// When set to true, helper threads terminate their work
static bool done_tracing{false};

// Unique buffer id in creation order
static std::atomic<uint64_t> buf_id{0};

// Unique id in buffer flush order
static std::atomic<uint64_t> flush_id{0};

static uint64_t get_and_inc_buf_id() { return buf_id++; }
static uint64_t get_buf_id() { return buf_id; }

static uint64_t get_and_inc_flush_id() { return flush_id++; }
static uint64_t get_flush_id() { return flush_id; }

/*
 * Used by OpenMP threads for assigning space for a trace record. If
 * there is no space in the last buffer allocated, the last buffer is
 * marked full and scheduled for flushing. Otherwise, space is
 * assigned for a trace record and the new cursor returned.
 */
void *OmptTracingBufferMgr::assignCursor(ompt_callbacks_t type) {
  // TODO Currently, we are serializing assignment of space for new
  // trace records as well as allocation of new buffers. This can be
  // changed by maintaining thread local info
  size_t rec_size = getTRSize();
  void *to_be_flushed_cursor = nullptr;
  std::unique_lock<std::mutex> lck(BufferMgrMutex);
  if (!Id2BufferMap.empty()) {
    // Try to assign a trace record from the last allocated buffer
    BufPtr buf = Id2BufferMap.rbegin()->second;
    if (rec_size <= buf->RemainingBytes) {
      // This is the new cursor
      assert((char *)buf->Start + buf->TotalBytes - buf->RemainingBytes ==
             (char *)buf->Cursor + rec_size);
      buf->Cursor = (char *)buf->Start + buf->TotalBytes - buf->RemainingBytes;
      buf->RemainingBytes -= rec_size;
      assert(Cursor2BufMdMap.find(buf->Cursor) == Cursor2BufMdMap.end());
      Cursor2BufMdMap[buf->Cursor] = std::make_shared<TraceRecordMd>(buf);
      DP("Assigned %lu bytes at %p in existing buffer %p\n", rec_size,
         buf->Cursor, buf->Start);
      return buf->Cursor;
    } else {
      to_be_flushed_cursor = buf->Cursor;
      buf->isFull = true; // no space for any more trace records
    }
  }
  void *buffer = nullptr;
  size_t total_bytes;
  // TODO Move the buffer allocation to a helper thread
  ompt_device_callbacks.ompt_callback_buffer_request(0 /* device_num */,
                                                     &buffer, &total_bytes);

  // TODO Instead of asserting, turn OFF tracing
  assert(total_bytes >= rec_size && "Buffer is too small");
  assert(buffer != nullptr && "Buffer request function failed");

  uint64_t new_buf_id = get_and_inc_buf_id();
  auto new_buf = std::make_shared<Buffer>(
      new_buf_id, buffer /* start */, buffer /* cursor */, total_bytes,
      total_bytes - rec_size, /* remaining bytes */
      false /* is full? */);
  auto buf_md_ptr = std::make_shared<TraceRecordMd>(new_buf);
  assert(Cursor2BufMdMap.find(new_buf->Cursor) == Cursor2BufMdMap.end());
  Cursor2BufMdMap[new_buf->Cursor] = buf_md_ptr;

  assert(Id2BufferMap.find(new_buf_id) == Id2BufferMap.end());
  Id2BufferMap[new_buf_id] = new_buf;

  lck.unlock();

  // Schedule this buffer for flushing till this cursor
  if (to_be_flushed_cursor)
    setComplete(to_be_flushed_cursor);

  DP("Assigned %lu bytes at %p in new buffer with id %lu\n", rec_size, buffer,
     new_buf_id);

  return buffer;
}

/*
 * Called by an OpenMP thread when a buffer fills up and should be
 * flushed. This function assigns a new flush_id to the buffer, adds
 * to the flush-related metadata and wakes up a helper thread to
 * dispatch a buffer-completion callback. This function should be
 * called without holding any lock.
 * Note lock order: buf_lock -> flush_lock
 */
void OmptTracingBufferMgr::setComplete(void *cursor) {
  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
  auto buf_itr = Cursor2BufMdMap.find(cursor);
  // Between calling setComplete and this check, a flush-all may have
  // delivered this buffer to the tool and deleted it. So the buffer
  // may not exist.
  if (buf_itr == Cursor2BufMdMap.end())
    return;

  // Cannot assert that the state of the cursor is ready since a
  // different thread may be in the process of populating it. If it
  // remains in init state when the range of trace records is
  // determined for dispatching the buffer-completion callback, it
  // will not be included.
  BufPtr buf = buf_itr->second->BufAddr;

  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  uint64_t flush_id;
  auto flush_itr = FlushBufPtr2IdMap.find(buf);
  if (flush_itr == FlushBufPtr2IdMap.end()) {
    // This buffer has not been flushed yet
    addNewFlushEntry(buf, cursor);
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
              ompt_device_callbacks.is_tracing_enabled()) ||
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
        dispatchCallback(flush_info.FlushBuf->Start, first_cursor, last_cursor);
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
    dispatchCallback(flush_info.FlushBuf->Start, first_cursor, last_cursor);
  }
}

// Given a range of trace records, dispatch a buffer-completion callback
void OmptTracingBufferMgr::dispatchCallback(void *buffer, void *first_cursor,
                                            void *last_cursor) {
  assert(first_cursor != nullptr && last_cursor != nullptr &&
         "Callback with nullptr");
  addLastCursor(last_cursor);

  // This is best effort.
  // There is a small window when the buffer-completion callback may
  // be invoked even after tracing has been disabled.
  // Note that we don't want to hold a lock when dispatching the callback.
  if (ompt_device_callbacks.is_tracing_enabled()) {
    DP("Dispatch callback w/ range (inclusive) to be flushed: %p -> %p\n",
       first_cursor, last_cursor);
    ompt_device_callbacks.ompt_callback_buffer_complete(
        0 /* TODO device num */, buffer,
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
  if (ompt_device_callbacks.is_tracing_enabled()) {
    DP("Dispatch callback with buffer %p owned\n", flush_info.FlushBuf->Start);
    ompt_device_callbacks.ompt_callback_buffer_complete(
        0, flush_info.FlushBuf->Start, 0, (ompt_buffer_cursor_t)0,
        true /* buffer owned */);
  }
}

void OmptTracingBufferMgr::setTRStatus(void *rec, TRStatus status) {
  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
  auto itr = Cursor2BufMdMap.find(rec);
  assert(itr != Cursor2BufMdMap.end());
  itr->second->TRState = status;
}

OmptTracingBufferMgr::TRStatus OmptTracingBufferMgr::getTRStatus(void *rec) {
  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
  auto itr = Cursor2BufMdMap.find(rec);
  assert(itr != Cursor2BufMdMap.end());
  return itr->second->TRState;
}

void *OmptTracingBufferMgr::getNextTR(void *rec) {
  size_t rec_size = getTRSize();
  // warning: no overflow check done
  return (char *)rec + rec_size;
}

bool OmptTracingBufferMgr::isBufferFull(const FlushInfo &flush_info) {
  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
  return flush_info.FlushBuf->isFull;
}

void *OmptTracingBufferMgr::getBufferCursor(BufPtr buf) {
  std::unique_lock<std::mutex> buf_lock(BufferMgrMutex);
  return buf->Cursor;
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
  // Mapping info for all cursors in this buffer must be erased. This
  // can be done since only fully populated buffers are destroyed.
  char *curr_cursor = (char*)flush_info.FlushBuf->Start;
  size_t total_valid_bytes = (buf->TotalBytes / getTRSize()) * getTRSize();
  char *end_cursor = curr_cursor + total_valid_bytes;
  while (curr_cursor != end_cursor) {
    auto buf_itr = Cursor2BufMdMap.find(curr_cursor);
    assert(buf_itr != Cursor2BufMdMap.end() &&
	   "Cursor not found in buffer metadata map");
    assert(buf_itr->second->BufAddr == buf);
    Cursor2BufMdMap.erase(buf_itr);
    curr_cursor += getTRSize();
  }
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
int OmptTracingBufferMgr::flushAllBuffers(ompt_device_t *device) {
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
    // If this buffer is in the flush-map, skip it. It is either in
    // process by another thread or will be processed
    BufPtr curr_buf = buf_itr->second;
    std::unique_lock<std::mutex> flush_lock(FlushMutex);
    auto flush_itr = FlushBufPtr2IdMap.find(curr_buf);
    if (flush_itr != FlushBufPtr2IdMap.end()) {
      ++curr_buf_id;
      continue;
    }
    // This buffer has not been flushed yet
    uint64_t flush_id = addNewFlushEntry(curr_buf, curr_buf->Cursor);
    DP("flushAllBuffers: Added new id %lu cursor %p buf %p\n", flush_id,
       curr_buf->Cursor, curr_buf->Start);

    flush_lock.unlock();
    buf_lock.unlock();

    ++curr_buf_id;
  }

  // Wake up all helper threads to invoke buffer-completion callbacks
  FlushCv.notify_all();

  // This is best effort. It is possible that some trace records are
  // not flushed when the wait is done.
  waitForFlushCompletion();

  return 1; // success
}

void OmptTracingBufferMgr::waitForFlushCompletion() {
  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  for (uint32_t i = 0; i < OMPT_NUM_HELPER_THREADS; ++i)
    setThreadFlush(i);
  ThreadFlushCv.wait(flush_lock, [this] { return ThreadFlushTracker == 0; });
}

void OmptTracingBufferMgr::init() {
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

void OmptTracingBufferMgr::shutdownHelperThreads() {
  std::unique_lock<std::mutex> flush_lock(FlushMutex);
  if (done_tracing // If another thread called stop, there is nothing
                   // to do for this thread
      || HelperThreadIdMap.empty() // Threads were never started
  ) {
    // Don't assert on HelperThreadIdMap since shutdown by another
    // thread may be in progress
    return;
  }

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

OmptTracingBufferMgr::~OmptTracingBufferMgr() {
  OMPT_TRACING_IF_ENABLED(shutdownHelperThreads(););
}
