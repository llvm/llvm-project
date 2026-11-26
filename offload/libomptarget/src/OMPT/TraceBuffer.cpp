//===- TraceBuffer.cpp - Generic device trace record buffering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OMPT/TraceBuffer.h"

#include "Shared/Debug.h"
#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <new>
#include <string>

using namespace llvm::omp::target::debug;

namespace llvm {
namespace omp {
namespace target {
namespace ompt {

namespace {
/// Next core instance id; ids are never reused.
std::atomic<uint64_t> NextInstanceId{1};

/// Core running a processing pass on this thread, if any.
thread_local const TraceBufferCore *ProcessingCore = nullptr;

#ifndef NDEBUG
/// Asserts that flushes and lifecycle calls never overlap, and that lifecycle
/// calls never overlap each other.
class CallTracker {
public:
  /// Counts a call in Mine; asserts Others idle and, if Exclusive, Mine too.
  CallTracker(std::atomic<int> &Mine, const std::atomic<int> &Others,
              bool Exclusive)
      : Mine(Mine) {
    int Prev = Mine.fetch_add(1);
    assert((!Exclusive || Prev == 0) && "Lifecycle calls do not overlap");
    assert(Others.load() == 0 && "Flushes and lifecycle calls do not overlap");
    (void)Prev;
  }
  /// Ends the counted call.
  ~CallTracker() { Mine.fetch_sub(1); }

private:
  std::atomic<int> &Mine;
};
#define TRACK_LIFECYCLE CallTracker Tracker(LifecycleCalls, FlushCalls, true)
#define TRACK_FLUSH CallTracker Tracker(FlushCalls, LifecycleCalls, false)
#else
#define TRACK_LIFECYCLE
#define TRACK_FLUSH
#endif
} // namespace

/// A provider buffer and its bump-allocation state. Only the thread whose
/// current buffer it is hands out slots from it.
struct TraceBufferCore::BufferInfoTy {
  uint64_t Id;
  int64_t DeviceId;
  void *Start;
  size_t TotalBytes;
  /// Bytes not yet handed out.
  size_t RemainingBytes;
  /// Last slot handed out.
  std::atomic<void *> Cursor;
  /// Set once no further slot fits; never cleared.
  std::atomic<bool> IsFull;

  /// Records a provider buffer and its allocation state.
  BufferInfoTy(uint64_t Id, int64_t DeviceId, void *Start, size_t TotalBytes,
               size_t RemainingBytes, void *Cursor, bool IsFull)
      : Id(Id), DeviceId(DeviceId), Start(Start), TotalBytes(TotalBytes),
        RemainingBytes(RemainingBytes), Cursor(Cursor), IsFull(IsFull) {}
  BufferInfoTy(const BufferInfoTy &) = delete;
  BufferInfoTy &operator=(const BufferInfoTy &) = delete;
};

TraceBufferCore::TraceBufferCore(LayoutTy Layout, BufferProvider &Provider,
                                 RecordConsumer &Consumer,
                                 TraceBufferConfig Config)
    : Layout(Layout), Provider(Provider), Consumer(Consumer), Config(Config),
      InstanceId(NextInstanceId.fetch_add(1, std::memory_order_relaxed)) {
  assert(Layout.Stride > 0 && "Slots have a size");
  assert(Layout.StatusOffset + sizeof(StatusTy) <= Layout.Stride &&
         "Status fits in a slot");
  assert(Layout.StatusOffset % alignof(StatusTy) == 0 &&
         Layout.Stride % alignof(StatusTy) == 0 &&
         "Every status is aligned in a malloc-aligned buffer");
  assert(Config.MaxDevices > 0 && "At least one device");
  assert(Config.HelperPollInterval.count() >= 0 &&
         "Poll interval not negative");
  assert(InstanceId != 0 &&
         InstanceId < std::numeric_limits<uint64_t>::max() - 1 &&
         "Instance id is a valid DenseMap key");
}

TraceBufferCore::~TraceBufferCore() {
  assert(ProcessingCore != this && "Not destroyed from its own callback");
  TRACK_LIFECYCLE;
#ifndef NDEBUG
  {
    std::lock_guard<std::mutex> Lock(FlushMutex);
    assert(!HelperThread.joinable() && "Helper still running at destruction");
  }
  if (Config.FlushOnShutdown) {
    std::lock_guard<std::mutex> Lock(BufferMgrMutex);
    for (auto &Entry : Id2BufferMap) {
      BufferInfoTy &Buf = *Entry.second;
      void *Curr = Buf.Start;
      void *Last = Buf.Cursor.load(std::memory_order_acquire);
      while (Curr <= Last) {
        assert(getSlotState(Curr) != SlotState::Ready &&
               "Unconsumed record at shutdown");
        Curr = nextSlot(Curr);
      }
    }
  }
#endif
  stopHelperImpl(/*FinalPass=*/false);
}

TraceBufferCore::SlotState TraceBufferCore::getSlotState(void *Slot) const {
  return getStatus(Slot).load(std::memory_order_acquire);
}

void TraceBufferCore::setSlotState(void *Slot, SlotState State) {
  getStatus(Slot).store(State, std::memory_order_release);
}

void TraceBufferCore::initSlotStatus(void *Slot) {
  assert(reinterpret_cast<uintptr_t>(&getStatus(Slot)) % alignof(StatusTy) ==
             0 &&
         "Slot status is aligned");
  new (&getStatus(Slot)) StatusTy(SlotState::Init);
}

bool TraceBufferCore::isSlotOf(const BufferInfoTy &Buf,
                               const void *Slot) const {
  const char *S = static_cast<const char *>(Slot);
  const char *Start = static_cast<const char *>(Buf.Start);
  return S >= Start && S + Layout.Stride <= Start + Buf.TotalBytes &&
         (S - Start) % Layout.Stride == 0;
}

std::vector<TraceBufferCore::BufPtr> &TraceBufferCore::getThreadBuffers() {
  thread_local llvm::DenseMap<uint64_t, std::vector<BufPtr>> PerCore;
  return PerCore[InstanceId];
}

void *TraceBufferCore::allocSlot(int64_t DeviceId) {
  if (DeviceId < 0 || DeviceId >= Config.MaxDevices)
    return nullptr;

  size_t RecSize = Layout.Stride;

  auto &Buffers = getThreadBuffers();
  if (Buffers.empty())
    Buffers.resize(Config.MaxDevices);
  BufPtr &Current = Buffers[DeviceId];

  if (Current != nullptr) {
    assert(Current->DeviceId == DeviceId && "Unexpected device id in buffer");
    void *OldCursor = Current->Cursor.load(std::memory_order_acquire);
    if (RecSize <= Current->RemainingBytes) {
      assert(!Current->IsFull.load(std::memory_order_relaxed) &&
             "A full buffer is never extended");
      assert(static_cast<char *>(Current->Start) + Current->TotalBytes -
                 Current->RemainingBytes ==
             static_cast<char *>(OldCursor) + RecSize);
      Current->RemainingBytes -= RecSize;

      void *NewCursor = static_cast<char *>(OldCursor) + RecSize;
      assert(isSlotOf(*Current, NewCursor) && "Cursor stays in its buffer");
      initSlotStatus(NewCursor);
      Current->Cursor.store(NewCursor, std::memory_order_release);
      return NewCursor;
    }
    assert(static_cast<char *>(OldCursor) + 2 * RecSize >
               static_cast<char *>(Current->Start) + Current->TotalBytes &&
           "Only a buffer without room becomes full");
    Current->IsFull.store(true, std::memory_order_release);
    // Queue before requesting a replacement, which the provider may decline.
    // Once queued the buffer may retire, so it is no longer current.
    if (Config.FlushOnBufferFull)
      triggerFlushOnBufferFull(OldCursor, Current);
    Current = nullptr;
  }

  BufferProvider::BufferTy NewBuf = Provider.allocateBuffer(DeviceId);
  if (NewBuf.Ptr == nullptr || NewBuf.Bytes < RecSize) {
    ODBG(ODT_Tool) << "Provider declined a buffer for device " << DeviceId;
    return nullptr;
  }
  assert(reinterpret_cast<uintptr_t>(NewBuf.Ptr) % alignof(std::max_align_t) ==
             0 &&
         "Provider buffer is not malloc-aligned");

  uint64_t NewBufId = BufId++;
  ODBG(ODT_Tool) << "New buffer " << NewBufId << " for device " << DeviceId;
  auto NewBufPtr = std::make_shared<BufferInfoTy>(
      NewBufId, DeviceId, /*Start=*/NewBuf.Ptr, NewBuf.Bytes,
      /*RemainingBytes=*/NewBuf.Bytes - RecSize, /*Cursor=*/NewBuf.Ptr,
      /*IsFull=*/false);
  assert(NewBufPtr->TotalBytes - NewBufPtr->RemainingBytes == RecSize &&
         "A new buffer hands out its first slot");

  initSlotStatus(NewBufPtr->Cursor.load(std::memory_order_acquire));
  Current = NewBufPtr;

  {
    std::lock_guard<std::mutex> Lock(BufferMgrMutex);
    Id2BufferMap[NewBufId] = NewBufPtr;
  }

  return NewBuf.Ptr;
}

void TraceBufferCore::publishSlot(void *Slot) {
  assert(reinterpret_cast<uintptr_t>(&getStatus(Slot)) % alignof(StatusTy) ==
             0 &&
         "Slot status is aligned");
  assert(getSlotState(Slot) == SlotState::Init && "Published at most once");
  setSlotState(Slot, SlotState::Ready);
}

void TraceBufferCore::triggerFlushOnBufferFull(void *Cursor, BufPtr Buf) {
  assert(Buf->IsFull.load(std::memory_order_acquire) &&
         Cursor == getBufferCursor(Buf) &&
         "Queued on full at its final cursor");
  std::unique_lock<std::mutex> BufLock(BufferMgrMutex);
  if (Id2BufferMap.find(Buf->Id) == Id2BufferMap.end())
    return;

  std::unique_lock<std::mutex> FlushLock(FlushMutex);
  auto It = FlushBufPtr2IdMap.find(Buf);
  if (It == FlushBufPtr2IdMap.end()) {
    addNewFlushEntry(Buf, Cursor);
    ODBG(ODT_Tool) << "Full buffer " << Buf->Id << " queued for flush";
  } else {
    auto MdIt = Id2FlushMdMap.find(It->second);
    assert(MdIt != Id2FlushMdMap.end());
    MdIt->second.FlushCursor = Cursor;
  }
  FlushLock.unlock();
  BufLock.unlock();

  FlushCv.notify_one();
}

uint64_t TraceBufferCore::addNewFlushEntry(BufPtr Buf, void *Cursor) {
  assert(FlushBufPtr2IdMap.find(Buf) == FlushBufPtr2IdMap.end());
  assert(isSlotOf(*Buf, Cursor) && "Flush cursor is a slot of its buffer");
  uint64_t Id = FlushId++;
  assert((Id2FlushMdMap.empty() || Id2FlushMdMap.rbegin()->first < Id) &&
         "Flush ids increase");
  FlushBufPtr2IdMap.emplace(Buf, Id);
  assert(Id2FlushMdMap.find(Id) == Id2FlushMdMap.end());
  Id2FlushMdMap.emplace(Id, FlushMd{Cursor, Buf, BufferFlushStatus::Waiting});
  ++QueuedGen;
  return Id;
}

void TraceBufferCore::queuePendingBuffers(int64_t DeviceId, bool AllDevices) {
  // To avoid holding the mutex for too long, get the first and last buffer
  // ids under the lock, then walk that range holding it per buffer.
  std::unique_lock<std::mutex> BufLock(BufferMgrMutex);
  if (Id2BufferMap.empty())
    return;
  uint64_t CurrId = Id2BufferMap.begin()->first;
  uint64_t LastId = Id2BufferMap.rbegin()->first;
  BufLock.unlock();

  while (CurrId <= LastId) {
    std::lock_guard<std::mutex> Lock(BufferMgrMutex);
    // Another thread may have retired this buffer by now.
    auto It = Id2BufferMap.find(CurrId);
    if (It == Id2BufferMap.end()) {
      ++CurrId;
      continue;
    }
    BufPtr Buf = It->second;
    assert(Buf->Id == CurrId && "Map key is the buffer id");

    if (!AllDevices && Buf->DeviceId != DeviceId) {
      ++CurrId;
      continue;
    }

    std::lock_guard<std::mutex> FlushLock(FlushMutex);
    if (FlushBufPtr2IdMap.find(Buf) == FlushBufPtr2IdMap.end())
      addNewFlushEntry(Buf, getBufferCursor(Buf));

    ++CurrId;
  }
}

TraceBufferCore::FlushInfo
TraceBufferCore::findAndReserveFlushedBuf(uint64_t Id) {
  std::lock_guard<std::mutex> Lock(FlushMutex);
  constexpr uint64_t MaxId = std::numeric_limits<uint64_t>::max();
  std::map<uint64_t, FlushMd>::iterator It;
  if (Id == MaxId) {
    if (Id2FlushMdMap.empty())
      return FlushInfo();
    for (It = Id2FlushMdMap.begin(); It != Id2FlushMdMap.end(); ++It)
      if (It->second.FlushStatus == BufferFlushStatus::Waiting)
        break;
    assert(It == Id2FlushMdMap.begin() &&
           "One processor at a time: nothing is mid-processing");
    if (It == Id2FlushMdMap.end())
      return FlushInfo();
  } else {
    It = Id2FlushMdMap.find(Id);
    assert((It == Id2FlushMdMap.end() ||
            It->second.FlushStatus != BufferFlushStatus::Processing) &&
           "One processor at a time: nothing is mid-processing");
    if (It == Id2FlushMdMap.end() ||
        It->second.FlushStatus == BufferFlushStatus::Processing)
      return FlushInfo();
  }
  assert(It->second.FlushStatus == BufferFlushStatus::Waiting);
  It->second.FlushStatus = BufferFlushStatus::Processing;
  It->second.FlushCursor =
      It->second.FlushBuf->Cursor.load(std::memory_order_acquire);
  assert(isSlotOf(*It->second.FlushBuf, It->second.FlushCursor) &&
         "Flush cursor is a slot of its buffer");
  return FlushInfo{It->first, It->second.FlushCursor, It->second.FlushBuf};
}

void TraceBufferCore::unreserveFlushedBuf(const FlushInfo &Info) {
  std::lock_guard<std::mutex> Lock(FlushMutex);
  auto It = Id2FlushMdMap.find(Info.FlushId);
  assert(It != Id2FlushMdMap.end() &&
         It->second.FlushStatus == BufferFlushStatus::Processing);
  It->second.FlushStatus = BufferFlushStatus::Waiting;
}

void TraceBufferCore::destroyFlushedBuf(const FlushInfo &Info) {
  BufPtr Buf = Info.FlushBuf;

  assert(Buf->IsFull.load(std::memory_order_acquire) &&
         "Only full buffers retire");
  std::lock_guard<std::mutex> BufLock(BufferMgrMutex);
  size_t Erased = Id2BufferMap.erase(Buf->Id);
  assert(Erased == 1 && "A retired buffer was registered");
  (void)Erased;

  std::lock_guard<std::mutex> FlushLock(FlushMutex);
  auto It = Id2FlushMdMap.find(Info.FlushId);
  assert(It != Id2FlushMdMap.end());
  assert(It->second.FlushBuf == Buf);
  assert(It->second.FlushStatus == BufferFlushStatus::Processing &&
         "Retired by its processor");
  Id2FlushMdMap.erase(It);
  FlushBufPtr2IdMap.erase(Buf);
}

void *TraceBufferCore::getBufferCursor(const BufPtr &Buf) const {
  return Buf->Cursor.load(std::memory_order_acquire);
}

bool TraceBufferCore::isBufferFull(const FlushInfo &Info) {
  std::lock_guard<std::mutex> Lock(BufferMgrMutex);
  return Info.FlushBuf->IsFull;
}

bool TraceBufferCore::isBufferOwned(const FlushInfo &Info) {
  assert(isBufferFull(Info) && "Compute buffer-owned when it is full");
  void *Curr = Info.FlushBuf->Start;
  void *Last = getBufferCursor(Info.FlushBuf);
  assert(static_cast<char *>(Last) + 2 * Layout.Stride >
             static_cast<char *>(Info.FlushBuf->Start) +
                 Info.FlushBuf->TotalBytes &&
         "A full buffer has no room for another slot");
  while (Curr <= Last) {
    if (getSlotState(Curr) != SlotState::Released)
      return false;
    Curr = nextSlot(Curr);
  }
  return true;
}

bool TraceBufferCore::flushBuffer(FlushInfo Info) {
  assert(Info.FlushBuf && "Cannot flush an empty buffer");
  assert(Info.FlushCursor && "Cannot flush upto a null cursor");

  void *Curr = Info.FlushBuf->Start;
  void *Last = Info.FlushCursor;
  assert(isSlotOf(*Info.FlushBuf, Last) &&
         "Flush cursor is a slot of its buffer");
  void *First = nullptr;
  void *LastReady = nullptr;
  bool Dispatched = false;
  while (Curr <= Last) {
    SlotState State = getSlotState(Curr);
    assert((State == SlotState::Init || State == SlotState::Ready ||
            State == SlotState::Released) &&
           "Slot status was initialised");
    if (State == SlotState::Init || State == SlotState::Released) {
      if (First == nullptr) {
        assert(LastReady == nullptr);
      } else {
        dispatchCallback(Info.FlushBuf->DeviceId, Info.FlushBuf->Start, First,
                         LastReady);
        Dispatched = true;
        First = LastReady = nullptr;
      }
    } else {
      assert(State == SlotState::Ready && "Unknown slot state");
      setSlotState(Curr, SlotState::Released);
      if (First == nullptr)
        First = Curr;
      LastReady = Curr;
    }
    Curr = nextSlot(Curr);
  }
  if (First != nullptr) {
    assert(LastReady != nullptr);
    dispatchCallback(Info.FlushBuf->DeviceId, Info.FlushBuf->Start, First,
                     LastReady);
    Dispatched = true;
  }
  return Dispatched;
}

void TraceBufferCore::dispatchCallback(int64_t DeviceId, void *Buffer,
                                       void *First, void *Last) {
  assert(First != nullptr && Last != nullptr);
  assert(First <= Last && "A run is ordered");
  assert(static_cast<char *>(First) >= static_cast<char *>(Buffer) &&
         (static_cast<char *>(First) - static_cast<char *>(Buffer)) %
                 Layout.Stride ==
             0 &&
         "A run starts at a slot of its buffer");
  size_t Bytes =
      static_cast<char *>(nextSlot(Last)) - static_cast<char *>(First);
  assert(Bytes > 0 && Bytes % Layout.Stride == 0 && "A run is whole slots");
  Consumer.consumeRecords(DeviceId, Buffer, First, Bytes);
}

void TraceBufferCore::dispatchBufferOwnedCallback(const FlushInfo &Info) {
  Provider.releaseBuffer(Info.FlushBuf->DeviceId, Info.FlushBuf->Start);
}

bool TraceBufferCore::invokeCallbacks() {
  constexpr uint64_t MaxId = std::numeric_limits<uint64_t>::max();
  uint64_t CurrId = MaxId;
  uint64_t EndId = FlushId.load();
  bool DidWork = false;

  while (true) {
    FlushInfo Info = findAndReserveFlushedBuf(CurrId);

    if (CurrId == MaxId && Info.FlushCursor == nullptr)
      return DidWork;

    if (Info.FlushCursor != nullptr)
      CurrId = Info.FlushId + 1;
    else {
      assert(CurrId != MaxId && "Cannot increment max id");
      ++CurrId;
    }

    if (Info.FlushCursor == nullptr) {
      if (CurrId < EndId)
        continue;
      return DidWork;
    }

    if (flushBuffer(Info))
      DidWork = true;

    if (isBufferFull(Info)) {
      if (isBufferOwned(Info)) {
        ODBG(ODT_Tool) << "Buffer " << Info.FlushBuf->Id
                       << " retired and released";
        destroyFlushedBuf(Info);
        dispatchBufferOwnedCallback(Info);
        DidWork = true;
      } else {
        unreserveFlushedBuf(Info);
      }
    } else {
      unreserveFlushedBuf(Info);
    }

    if (CurrId >= EndId)
      return DidWork;
  }
}

bool TraceBufferCore::processPending() {
  assert(ProcessingCore != this && "Not re-entered from its own callback");
  std::lock_guard<std::mutex> Lock(ProcessMutex);
  const TraceBufferCore *Prev = ProcessingCore;
  ProcessingCore = this;
  bool DidWork = invokeCallbacks();
  ProcessingCore = Prev;
  return DidWork;
}

void TraceBufferCore::setEnabled(bool NewEnabled) {
  {
    std::lock_guard<std::mutex> Lock(FlushMutex);
    Enabled.store(NewEnabled, std::memory_order_release);
  }
  FlushCv.notify_all();
}

void TraceBufferCore::bumpAndWaitForCompletion(
    std::unique_lock<std::mutex> &Lock) {
  assert(CompletedGen <= RequestedGen && "Completion never runs ahead");
  uint64_t MyGen = ++RequestedGen;
  Lock.unlock();
  FlushCv.notify_all();
  Lock.lock();
  ThreadFlushCv.wait(Lock, [this, MyGen] { return CompletedGen >= MyGen; });
}

bool TraceBufferCore::doFlush(int64_t DeviceId, bool AllDevices) {
  if (ProcessingCore == this)
    return false;

  ODBG(ODT_Tool) << "Flush "
                 << (AllDevices ? "all devices"
                                : "device " + std::to_string(DeviceId))
                 << ": start";
  queuePendingBuffers(DeviceId, AllDevices);

  std::unique_lock<std::mutex> Lock(FlushMutex);
  if (!HelperThread.joinable()) {
    Lock.unlock();
    processPending();
  } else {
    bumpAndWaitForCompletion(Lock);
  }
  ODBG(ODT_Tool) << "Flush "
                 << (AllDevices ? "all devices"
                                : "device " + std::to_string(DeviceId))
                 << ": end";
  return true;
}

bool TraceBufferCore::flush(int64_t DeviceId) {
  if (DeviceId < 0 || DeviceId >= Config.MaxDevices)
    return false;
  if (ProcessingCore == this)
    return false;
  TRACK_FLUSH;
  return doFlush(DeviceId, /*AllDevices=*/false);
}

bool TraceBufferCore::flushAll() {
  if (ProcessingCore == this)
    return false;
  TRACK_FLUSH;
  return doFlush(0, /*AllDevices=*/true);
}

void TraceBufferCore::driveCompletion() {
  bool RunFinalPass = false;
  while (true) {
    std::unique_lock<std::mutex> Lock(FlushMutex);
    if (Done) {
      RunFinalPass = FinalPassOnStop;
      break;
    }

    bool QueuedWork = !Id2FlushMdMap.empty() && Enabled.load();
    if (QueuedWork) {
      uint64_t SeenQueuedGen = QueuedGen;
      FlushCv.wait_for(Lock, Config.HelperPollInterval, [this, SeenQueuedGen] {
        return Done || RequestedGen > CompletedGen ||
               QueuedGen != SeenQueuedGen;
      });
    } else {
      FlushCv.wait(Lock, [this] {
        return Done || RequestedGen > CompletedGen ||
               (!Id2FlushMdMap.empty() && Enabled.load());
      });
    }
    if (Done) {
      RunFinalPass = FinalPassOnStop;
      break;
    }
    if (RequestedGen <= CompletedGen && !Enabled.load())
      continue;

    uint64_t Gen = RequestedGen;
    Lock.unlock();

    processPending();

    Lock.lock();
    assert(Gen <= RequestedGen && "Completes only requested generations");
    CompletedGen = std::max(CompletedGen, Gen);
    assert(CompletedGen <= RequestedGen && "Completion never runs ahead");
    Lock.unlock();
    ThreadFlushCv.notify_all();
  }

  if (RunFinalPass)
    processPending();
}

void TraceBufferCore::startHelper() {
  assert(ProcessingCore != this && "Not called from its own callback");
  TRACK_LIFECYCLE;
  std::lock_guard<std::mutex> Lock(FlushMutex);
  if (HelperThread.joinable())
    return;
  Done = false;
  HelperThread = std::thread(&TraceBufferCore::driveCompletion, this);
  ODBG(ODT_Tool) << "Helper thread started";
}

void TraceBufferCore::stopHelperImpl(bool FinalPass) {
  std::unique_lock<std::mutex> Lock(FlushMutex);
  if (!HelperThread.joinable())
    return;
  assert(std::this_thread::get_id() != HelperThread.get_id() &&
         "The helper does not stop itself");
  Done = true;
  FinalPassOnStop = FinalPass;
  Lock.unlock();
  FlushCv.notify_all();
  HelperThread.join();
  ODBG(ODT_Tool) << "Helper thread stopped";
}

void TraceBufferCore::stopHelper() {
  assert(ProcessingCore != this && "Not called from its own callback");
  TRACK_LIFECYCLE;
  stopHelperImpl(/*FinalPass=*/true);
}

void TraceBufferCore::flushAndStopHelper() {
  assert(ProcessingCore != this && "Not called from its own callback");
  TRACK_LIFECYCLE;
  if (Config.FlushOnShutdown) {
    doFlush(0, /*AllDevices=*/true);
  } else {
    std::unique_lock<std::mutex> Lock(FlushMutex);
    if (HelperThread.joinable())
      bumpAndWaitForCompletion(Lock);
  }
  stopHelperImpl(/*FinalPass=*/true);
}

} // namespace ompt
} // namespace target
} // namespace omp
} // namespace llvm
