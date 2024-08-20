//===- StaticMemoryPlanning.cpp - Static memory planning ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/StaticMemoryPlanning.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace mlir {
namespace memoryplan {
namespace {
static constexpr size_t divideAndCeil(size_t x, size_t y) {
  return (x + y - 1) / y;
}
// how the buffer was created
enum class ChunkType {
  ORIGIN, // the chunk is directly allocated from the large buffer
  SPLIT,  // the chunk is got by splitting another memory chunk
  MERGED, // the chunk is got by merging several consecutive memory chunks
};

struct MemoryState;

struct MemoryChunk {
  ChunkType type;
  size_t size;
  bool isfree = true;
  size_t lastFreedTick = 0;
  bool isInplaceSplitRemainder = false;
  // splits the chunk and get the left hand side with size = size, registers
  // both the returned chunk and the rest of the chunk to the state
  void split(MemoryState *state, size_t size, MemoryChunk *&lhs,
             MemoryChunk *&rhs);
  // move the buffer, propagate the message up to the parent chunk. It will
  // not update the siblings.
  virtual void move(int64_t startDiff) = 0;
  // extend the buffer, propagate the message up to the parent chunk. It will
  // not update the siblings.
  virtual void extend(int64_t sizeDiff) = 0;

  MemoryChunk(ChunkType type, size_t size) : type(type), size(size) {}
  // there should be no updates to memory chunks after calling
  // getStartOffset
  size_t getStartOffset() {
    if (cached_start_offset == UNINITIALIZED) {
      cached_start_offset = getStartOffsetImpl();
    }
    return cached_start_offset;
  }
  virtual ~MemoryChunk() = default;

  virtual size_t getStartOffsetImpl() = 0;

protected:
  static constexpr size_t UNINITIALIZED = std::numeric_limits<size_t>::max();
  size_t cached_start_offset = UNINITIALIZED;
};

// the memory chunk that is directly allocated from the large buffer
struct OriginChunk : public MemoryChunk {
  // no parent
  // MemoryChunk *parent;
  size_t start;
  OriginChunk(size_t start, size_t size)
      : MemoryChunk{ChunkType::ORIGIN, size}, start(start) {}
  void move(int64_t startDiff) override { start += startDiff; }
  void extend(int64_t sizeDiff) override { size += sizeDiff; }
  size_t getStartOffsetImpl() override { return start; }
};

// the memory chunk that is split from another chunk
struct SplitChunk : public MemoryChunk {
  MemoryChunk *parent;
  // if the chunk is the left hand side (smaller starting offset)
  bool isLHS;
  SplitChunk(size_t size, MemoryChunk *parent, bool is_lhs)
      : MemoryChunk{ChunkType::SPLIT, size}, parent(parent), isLHS(is_lhs) {}
  void move(int64_t startDiff) override {
    if (isLHS) {
      parent->move(startDiff);
    }
    // no need to pass message to parent for rhs, since lhs has done so
  }
  void extend(int64_t sizeDiff) override {
    size += sizeDiff;
    parent->extend(sizeDiff);
    // if is_lhs, we will later call rhs->move(...)
  }
  size_t getStartOffsetImpl() override {
    if (isLHS) {
      return parent->getStartOffset();
    } else {
      return parent->getStartOffset() + parent->size - size;
    }
  }
};

static size_t getSizeOfChunks(const std::vector<MemoryChunk *> &c) {
  size_t v = 0;
  for (auto chk : c) {
    v += chk->size;
  }
  return v;
}
// the memory chunk that is merged from another chunks
struct MergedChunk : public MemoryChunk {
  std::vector<MemoryChunk *> parent;
  MergedChunk(std::vector<MemoryChunk *> &&parent)
      : MemoryChunk{ChunkType::MERGED, getSizeOfChunks(parent)},
        parent(std::move(parent)) {}
  void move(int64_t startDiff) override {
    for (auto v : parent) {
      v->move(startDiff);
    }
  }
  void extend(int64_t sizeDiff) override {
    size += sizeDiff;
    parent.back()->extend(sizeDiff);
  }
  size_t getStartOffsetImpl() override {
    return parent.front()->getStartOffset();
  }
};

struct MemoryState {
  // buffer_id -> allocated memory chunk, used to collect the final result
  std::unordered_map<uintptr_t, MemoryChunk *> allocations;
  // buffer_id -> <current_alive_memory_chunk>, used when inplace
  // optimization, and when a buffer is inplace reused by another buffer. The
  // reused buffer will have unchanged MemoryChunk in allocations, because
  // allocations shows the final result of the buffer. curAllocations
  // tracks the current mapping of buffer_id to MemoryChunk, which may be
  // different from allocations
  std::unordered_map<uintptr_t, MemoryChunk *> curAllocations;
  // all memory chunks that has been created, takes the ownerships of the
  // MemoryChunk objects
  std::vector<std::unique_ptr<MemoryChunk>> chunks;
  // the current memory chunks, sorted by the starting offset
  std::vector<MemoryChunk *> curChunks;
  // free chunks sorted by size
  std::multimap<size_t, MemoryChunk *> freeChunksBySize;
  // free chunks sorted by last freed tick
  std::multimap<size_t, MemoryChunk *> freeChunksByTick;
  // the current size of the large buffer, in number of elements
  size_t currentAllocSize = 0;
  // the alignment in number of elements
  size_t alignment;
  // the map from a buffer-id to the buffer-ids that the buffer can inplace
  // reuse
  const InplaceInfoMap &inplaceMap;
  std::unordered_map<uintptr_t, std::vector<uintptr_t>> &outInplaceSelection;
  int tick = 0;
  bool hotFirst;

  MemoryState(size_t alignment, bool hotFirst, const InplaceInfoMap &inplaceMap,
              std::unordered_map<uintptr_t, std::vector<uintptr_t>>
                  &outInplaceSelection)
      : alignment(alignment), inplaceMap(inplaceMap),
        outInplaceSelection(outInplaceSelection), hotFirst(hotFirst) {}

  void removeChunkFromMap(MemoryChunk *target, size_t t,
                          std::multimap<size_t, MemoryChunk *> &m) {
    auto mapitr = m.equal_range(t);
    assert(mapitr.first != mapitr.second);
    for (auto map_i = mapitr.first; map_i != mapitr.second; ++map_i) {
      if (map_i->second == target) {
        m.erase(map_i);
        break;
      }
    }
  }
  void removeChunkFromFreeList(MemoryChunk *target) {
    removeChunkFromMap(target, target->size, freeChunksBySize);
    removeChunkFromMap(target, target->lastFreedTick, freeChunksByTick);
    target->isfree = false;
  }

  void addChunkToFreeList(MemoryChunk *target) {
    freeChunksBySize.insert(std::make_pair(target->size, target));
    freeChunksByTick.insert(std::make_pair(target->lastFreedTick, target));
    target->isfree = true;
  }

  void extendAlloc(MemoryChunk *target, size_t aligned) {
    // remove the chunk from free list
    removeChunkFromFreeList(target);
    int64_t sizeDiff = aligned - target->size;
    assert(sizeDiff > 0);
    currentAllocSize += sizeDiff;
    // extend the target chunk, also move all buffers at the right of it
    target->extend(sizeDiff);
    bool found_target = false;
    for (auto v : curChunks) {
      if (v == target) {
        found_target = true;
      } else if (found_target) {
        // v is at the right of the target
        v->move(sizeDiff);
      }
    }
    assert(found_target);
    target->isfree = false;
  }

  MemoryChunk *splitAlloc(MemoryChunk *target, size_t aligned,
                          MemoryChunk *&rhs_ret) {
    // found a free chunk that is large enough
    if (target->size == aligned) {
      // a perfect match, no need to split
      auto ret = target;
      removeChunkFromFreeList(target);
      return ret;
    }
    // split the larger chunk
    assert(target->size > aligned);
    auto lhs = std::make_unique<SplitChunk>(aligned, target, true);
    auto rhs =
        std::make_unique<SplitChunk>(target->size - aligned, target, false);
    rhs_ret = rhs.get();
    auto ret = lhs.get();

    auto oldItrInCurChunks =
        std::find(curChunks.begin(), curChunks.end(), target);
    assert(oldItrInCurChunks != curChunks.end());
    // replace old chunk with rhs
    *oldItrInCurChunks = rhs.get();
    // insert lhs before rhs
    curChunks.insert(oldItrInCurChunks, lhs.get());
    rhs->lastFreedTick = target->lastFreedTick;
    // add rhs to free list
    addChunkToFreeList(rhs.get());

    // move ownership
    chunks.emplace_back(std::move(lhs));
    chunks.emplace_back(std::move(rhs));

    // remove old chunk in free list
    removeChunkFromFreeList(target);
    ret->isfree = false;
    return ret;
  }

  float calculateSizeScore(size_t chkSize, size_t allocSize) const {
    // size_score = abs(chunk_size-allocSize)/max(chunk_size, allocSize)
    int64_t sizeDiff =
        static_cast<int64_t>(chkSize) - static_cast<int64_t>(allocSize);
    float size_max = static_cast<float>(std::max(allocSize, chkSize));
    float size_score = -std::abs(sizeDiff) / size_max;
    // if we don't need to extend the buffer, add a bounus score for it
    if (allocSize <= chkSize) {
      size_score += 1;
    }
    // size_score and tick_score are normalized in [-1,1]. We set a weight
    // for these two scores: 1:1
    return size_score;
  }

  // calculates the score of a free chunk to help select the best chunk we
  // allocate memory from. It considers 2 factors: 1) the free chunk size and
  // the size of the current memory allocation request. The closer they are,
  // the better the chunk is. 2) the heat of the chunk. If the chunk's last
  // free'd tick is closer to the current tick, the chunk is better.
  // The better the chunk is, the greater the score is
  float calculateChunkScore(MemoryChunk *chk, size_t allocSize,
                            size_t lastTick) const {
    // if the buffer is free'd N ticks ago, it will have score max(0, 1 - N
    // * 0.1)
    float tick_score = static_cast<float>(tick - lastTick) / 10;
    tick_score = 1 - std::min(tick_score, 1.0f);
    // size_score and tick_score are normalized in [-1,1]. We set a weight
    // for these two scores: 1:1
    return 1 * calculateSizeScore(chk->size, allocSize) + 1 * tick_score;
  }

  MemoryChunk *alloc(uintptr_t bufferid, size_t size) {
    tick++;
    auto ret = doAlloc(bufferid, size);
    allocations[bufferid] = ret;
    curAllocations[bufferid] = ret;
    return ret;
  }

  // check if the buffer is split from a base tensor and check the
  // InplaceInfo for whether it requires zero offset
  bool checkBufferOffsetForInplace(MemoryChunk *chunk,
                                   const InplaceInfo *info) {
    // if the old memory chunk is splitted from the base tensor
    bool oldIsSplit = chunk->isInplaceSplitRemainder;
    // if the old memory chunk is based on a offset of the base tensor
    // and we require that we should use zero offset on that tensor, we
    // cannot reuse it
    return !oldIsSplit || info->second != InplaceKind::ZERO_OFFSET;
  }

  // find the range of chunks in curChunks that can be merged for inplace
  // reuse, returns the memory size of the range and the start/end iterators
  size_t findInplaceMergeRange(
      MemoryChunk *victim, size_t aligned,
      const std::unordered_map<MemoryChunk *, const InplaceInfo *> &can_inplace,
      std::vector<MemoryChunk *>::iterator &toMergeStart,
      std::vector<MemoryChunk *>::iterator &toMergeEnd) {
    // addChunkToFreeList(chk);
    auto itrInCurChunks = std::find(curChunks.begin(), curChunks.end(), victim);
    assert(itrInCurChunks != curChunks.end());
    // merge right if they are free or can be inplaced
    toMergeStart = itrInCurChunks;
    toMergeEnd = itrInCurChunks + 1;
    // remember the memory size we already collected. If
    // currentCollectedSize is greater than the memory size to alloc, we
    // can stop searching
    size_t currentCollectedSize = victim->size;
    // look right to see any one we can merge with
    for (auto itr = itrInCurChunks + 1;
         itr != curChunks.end() && currentCollectedSize < aligned; ++itr) {
      // if the memory chunk is in use and is in can_inplace map, we may
      // reuse it now
      auto inplaceInfoItr = can_inplace.find(*itr);
      if ((*itr)->isfree ||
          (inplaceInfoItr != can_inplace.end() &&
           inplaceInfoItr->second->second == InplaceKind::FREE)) {
        toMergeEnd = itr + 1;
        currentCollectedSize += (*itr)->size;
      } else {
        break;
      }
    }
    return currentCollectedSize;
  }

  // inplace alloc memory on a chunk that is in use, but about to be freed.
  MemoryChunk *doInplaceAlloc(uintptr_t bufferid, size_t aligned) {
    if (inplaceMap.empty()) {
      return nullptr;
    }
    auto itr_inplace = inplaceMap.find(bufferid);
    if (itr_inplace == inplaceMap.end()) {
      return nullptr;
    }
    // if the buffer can inplace reuse some other buffers that is
    // still in use but about to be freed
    const auto &bufferCanInplace = itr_inplace->second;
    if (bufferCanInplace.empty()) {
      return nullptr;
    }

    // reversed map, chunk --> buffer id for inplace candidates
    std::unordered_map<MemoryChunk *, const InplaceInfo *> can_inplace;
    for (auto &v : bufferCanInplace) {
      auto itr = curAllocations.find(v.first);
      if (itr != curAllocations.end()) {
        can_inplace[itr->second] = &v;
      }
    }

    // stage 1, find a victim based on the memory size that can be freed
    float targetScore = -std::numeric_limits<float>::infinity();
    MemoryChunk *victim = nullptr;
    std::vector<MemoryChunk *>::iterator toMergeStart;
    std::vector<MemoryChunk *>::iterator toMergeEnd;
    size_t currentCollectedSize = 0;
    for (auto &bufinfo : bufferCanInplace) {
      auto buf_id = bufinfo.first;
      auto old_buf_itr = curAllocations.find(buf_id);
      // if the buffer has already been reused by other buffers, skip
      if (old_buf_itr == curAllocations.end()) {
        continue;
      }
      // the old memory chunk
      auto old_buf = old_buf_itr->second;

      auto &old_inplace_info = can_inplace[old_buf];
      if (!checkBufferOffsetForInplace(old_buf, old_inplace_info)) {
        continue;
      }

      std::vector<MemoryChunk *>::iterator curMergeStart;
      std::vector<MemoryChunk *>::iterator curMergeEnd;
      auto cur_size = findInplaceMergeRange(old_buf, aligned, can_inplace,
                                            curMergeStart, curMergeEnd);
      float score = calculateSizeScore(cur_size, aligned);
      if (score > targetScore) {
        targetScore = score;
        victim = old_buf;
        toMergeStart = curMergeStart;
        toMergeEnd = curMergeEnd;
        currentCollectedSize = cur_size;
      }
    }
    if (currentCollectedSize * 10 < aligned) {
      // if the memory can be reused is too small (less than 10% of the
      // target size), inplacing has no benifits, skip
      return nullptr;
    }
    if (!victim) {
      return nullptr;
    }
    assert(!victim->isfree);

    victim->lastFreedTick = tick;

    std::vector<MemoryChunk *> merged_buffers(toMergeStart, toMergeEnd);
    for (auto buf : merged_buffers) {
      auto itr = can_inplace.find(buf);
      if (itr != can_inplace.end()) {
        uintptr_t vic_buffer_id = itr->second->first;
        if (vic_buffer_id) {
          outInplaceSelection[bufferid].emplace_back(vic_buffer_id);
          DEBUG_WITH_TYPE("memplan", llvm::dbgs() << "Buffer " << bufferid
                                                  << " inplace reuses "
                                                  << vic_buffer_id << "\n");
        }
      }
    }
    if (currentCollectedSize < aligned) {
      // if the collected memory size is still less than the size to
      // alloc, need to extend
      auto targetSize =
          aligned - currentCollectedSize + merged_buffers.back()->size;
      if (!merged_buffers.back()->isfree) {
        // if it is not free, we are inplacing it. Temporarily move to
        // free list
        addChunkToFreeList(merged_buffers.back());
      }
      extendAlloc(merged_buffers.back(), targetSize);
      // after extension of the last buffer, the collected size is equal
      // to the size to alloc
      currentCollectedSize = aligned;
    }

    // remove from freelist and buffer_id->chunk map
    for (auto itr = toMergeStart; itr != toMergeEnd; ++itr) {
      auto chunk = *itr;
      if (chunk->isfree) {
        removeChunkFromFreeList(chunk);
      }
      auto itrChunk = can_inplace.find(chunk);
      if (itrChunk != can_inplace.end()) {
        curAllocations.erase(itrChunk->second->first);
      }
    }

    MemoryChunk *mergedChunk;
    // if we need to merge multiple chunks
    if (toMergeEnd - toMergeStart > 1) {
      // do merge
      chunks.emplace_back(std::make_unique<MergedChunk>(
          std::vector<MemoryChunk *>(merged_buffers)));
      mergedChunk = chunks.back().get();
      // remove merged chunks from free list and cur_chunk list
      // add merged chunk to cur_chunks and free_chunks_by_size
      *toMergeStart = mergedChunk;
      mergedChunk->lastFreedTick = tick;
      mergedChunk->isfree = false;
      curChunks.erase(toMergeStart + 1, toMergeEnd);
    } else {
      mergedChunk = victim;
      mergedChunk->lastFreedTick = tick;
    }

    // mergedChunk is in curChunks and is removed from freelist and
    // curAllocations map
    if (currentCollectedSize == aligned) {
      // if is extended, or perfect match, just return the chunk
      mergedChunk->isfree = false;
      return mergedChunk;
    } else {
      // otherwise, there are some unused memory in the chunk to be
      // reused. We need to split it. If the RHS of the chunk is from a
      // inplace reused buffer, need to add a mapping of the buffer id to
      // the RHS remaining chunk
      if (!mergedChunk->isfree) {
        addChunkToFreeList(mergedChunk);
      }
      MemoryChunk *rhs = nullptr;
      auto ret = splitAlloc(mergedChunk, aligned, rhs);
      auto itrChunk = can_inplace.find(merged_buffers.back());
      if (itrChunk != can_inplace.end()) {
        // if the last chunk is from inplace map, the RHS chunk is not
        // really freed, need to remove from free list and mark it not
        // freed.
        removeChunkFromFreeList(rhs);
        rhs->isInplaceSplitRemainder = true;
        // update the buffer id -> chunk map, so that when freeing the
        // inplaced buffer, we can find the correct remaining buffer
        curAllocations[itrChunk->second->first] = rhs;
      }
      return ret;
    }
  }

  MemoryChunk *doAlloc(uintptr_t bufferid, size_t size) {
    auto aligned = divideAndCeil(size, alignment) * alignment;
    // try inplace
    if (auto inp_ret = doInplaceAlloc(bufferid, size)) {
      return inp_ret;
    }
    if (freeChunksBySize.empty()) {
      chunks.emplace_back(
          std::make_unique<OriginChunk>(currentAllocSize, aligned));
      currentAllocSize += aligned;
      auto ret = chunks.back().get();
      curChunks.emplace_back(ret);
      ret->isfree = false;
      return ret;
    }
    if (hotFirst) {
      MemoryChunk *target = freeChunksByTick.rbegin()->second;
      float targetScore = calculateChunkScore(target, aligned,
                                              freeChunksByTick.rbegin()->first);
      for (auto &kv : freeChunksByTick) {
        float score = calculateChunkScore(kv.second, aligned, kv.first);
        if (score > targetScore) {
          target = kv.second;
          targetScore = score;
        }
      }
      if (target->size < aligned) {
        extendAlloc(target, aligned);
        return target;
      } else {
        MemoryChunk *rhs;
        return splitAlloc(target, aligned, rhs);
      }
    } else {
      // find a free chunk that best fits the current size
      // itr will be the smallest chunk whose size >= aligned
      auto itr = freeChunksBySize.lower_bound(aligned);
      if (itr == freeChunksBySize.end()) {
        MemoryChunk *target;
        // itr points to the last element
        --itr;
        // if not found, this means that all free chunk is smaller than
        // aligned size, switch to the largest chunk
        target = itr->second;
        extendAlloc(target, aligned);
        return target;
      } else {
        MemoryChunk *rhs;
        return splitAlloc(itr->second, aligned, rhs);
      }
    }
  }

  void dealloc(MemoryChunk *chk) {
    tick++;
    chk->lastFreedTick = tick;
    addChunkToFreeList(chk);
    auto itrInCurChunks = std::find(curChunks.begin(), curChunks.end(), chk);
    assert(itrInCurChunks != curChunks.end());
    // merge left and right if they are free
    std::vector<MemoryChunk *>::iterator toMergeStart = itrInCurChunks;
    std::vector<MemoryChunk *>::iterator toMergeEnd = itrInCurChunks + 1;
    // look left to see any one we can merge with
    for (auto itr = itrInCurChunks;; --itr) {
      if ((*itr)->isfree) {
        toMergeStart = itr;
      } else {
        break;
      }
      if (itr == curChunks.begin()) {
        break;
      }
    }
    // look right to see any one we can merge with
    for (auto itr = itrInCurChunks + 1; itr != curChunks.end(); ++itr) {
      if ((*itr)->isfree) {
        toMergeEnd = itr + 1;
      } else {
        break;
      }
    }
    if (toMergeEnd - toMergeStart > 1) {
      // do merge
      chunks.emplace_back(std::make_unique<MergedChunk>(
          std::vector<MemoryChunk *>(toMergeStart, toMergeEnd)));

      // remove merged chunks from free list and cur_chunk list
      for (auto itr = toMergeStart; itr != toMergeEnd; ++itr) {
        auto chunk = *itr;
        removeChunkFromFreeList(chunk);
      }
      // add merged chunk to cur_chunks and free_chunks_by_size
      *toMergeStart = chunks.back().get();
      chunks.back()->lastFreedTick = tick;
      addChunkToFreeList(chunks.back().get());
      curChunks.erase(toMergeStart + 1, toMergeEnd);
    }
    // else, no chunks are merged, do nothing
  }

  void dealloc(uintptr_t bufferid) {
    auto alocitr = allocations.find(bufferid);
    assert(alocitr != allocations.end() &&
           "Cannot find buffer id in allocations");
    auto itr = curAllocations.find(bufferid);
    if (itr != curAllocations.end()) {
      itr->second->isInplaceSplitRemainder = false;
      dealloc(itr->second);
      curAllocations.erase(itr);
    }
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "total size " << currentAllocSize << " ";
    size_t cur_offset = 0;
    for (auto buf : curChunks) {
      ss << "| " << cur_offset << ',' << buf->size << ',' << buf->isfree << " ";
      cur_offset += buf->size;
    }
    return ss.str();
  }
};
} // namespace

size_t scheduleMemoryAllocations(
    const Traces &traces, std::size_t alignment, bool hotFirst,
    const InplaceInfoMap &inplaceMap,
    std::unordered_map<uintptr_t, std::size_t> &outSchedule,
    std::unordered_map<uintptr_t, std::vector<uintptr_t>>
        &outInplaceSelection) {
  MemoryState planner{alignment, hotFirst, inplaceMap, outInplaceSelection};
  for (auto &trace : traces) {
    if (trace.size > 0) {
      planner.alloc(trace.bufferId, trace.size);
      DEBUG_WITH_TYPE("memplan", llvm::dbgs() << "Alloc " << trace.bufferId
                                              << ", sz=" << trace.size << "\n"
                                              << planner.toString() << "\n");
    } else {
      planner.dealloc(trace.bufferId);
      DEBUG_WITH_TYPE("memplan", llvm::dbgs()
                                     << "Dealloc " << trace.bufferId << "\n"
                                     << planner.toString() << "\n");
    }
  }
  for (auto &kv : planner.allocations) {
    outSchedule[kv.first] = kv.second->getStartOffset();
  }
  return planner.currentAllocSize;
}

} // namespace memoryplan
} // namespace mlir
