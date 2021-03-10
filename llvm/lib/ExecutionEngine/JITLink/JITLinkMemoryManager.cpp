//===--- JITLinkMemoryManager.cpp - JITLinkMemoryManager implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/Support/Process.h"

namespace llvm {
namespace jitlink {

JITLinkMemoryManager::~JITLinkMemoryManager() = default;
JITLinkMemoryManager::Allocation::~Allocation() = default;

Expected<std::unique_ptr<JITLinkMemoryManager::Allocation>>
InProcessMemoryManager::allocate(const JITLinkDylib *JD,
                                 const SegmentsRequestMap &Request) {

  using AllocationMap = DenseMap<unsigned, sys::MemoryBlock>;

  // Local class for allocation.
  class IPMMAlloc : public Allocation {
  public:
    IPMMAlloc(AllocationMap SegBlocks) : SegBlocks(std::move(SegBlocks)) {}
    MutableArrayRef<char> getWorkingMemory(ProtectionFlags Seg) override {
      assert(SegBlocks.count(Seg) && "No allocation for segment");
      return {static_cast<char *>(SegBlocks[Seg].base()),
              SegBlocks[Seg].allocatedSize()};
    }
    JITTargetAddress getTargetMemory(ProtectionFlags Seg) override {
      assert(SegBlocks.count(Seg) && "No allocation for segment");
      return pointerToJITTargetAddress(SegBlocks[Seg].base());
    }
    void finalizeAsync(FinalizeContinuation OnFinalize) override {
      OnFinalize(applyProtections());
    }
    Error deallocate() override {
      if (SegBlocks.empty())
        return Error::success();
      void *SlabStart = SegBlocks.begin()->second.base();
      char *SlabEnd = (char *)SlabStart;
      for (auto &KV : SegBlocks) {
        SlabStart = std::min(SlabStart, KV.second.base());
        SlabEnd = std::max(SlabEnd, (char *)(KV.second.base()) +
                                        KV.second.allocatedSize());
      }
      size_t SlabSize = SlabEnd - (char *)SlabStart;
      assert((SlabSize % sys::Process::getPageSizeEstimate()) == 0 &&
             "Slab size is not a multiple of page size");
      sys::MemoryBlock Slab(SlabStart, SlabSize);
      if (auto EC = sys::Memory::releaseMappedMemory(Slab))
        return errorCodeToError(EC);
      return Error::success();
    }

  private:
    Error applyProtections() {
      for (auto &KV : SegBlocks) {
        auto &Prot = KV.first;
        auto &Block = KV.second;
        if (auto EC = sys::Memory::protectMappedMemory(Block, Prot))
          return errorCodeToError(EC);
        if (Prot & sys::Memory::MF_EXEC)
          sys::Memory::InvalidateInstructionCache(Block.base(),
                                                  Block.allocatedSize());
      }
      return Error::success();
    }

    AllocationMap SegBlocks;
  };

  if (!isPowerOf2_64((uint64_t)sys::Process::getPageSizeEstimate()))
    return make_error<StringError>("Page size is not a power of 2",
                                   inconvertibleErrorCode());

  AllocationMap Blocks;
  const sys::Memory::ProtectionFlags ReadWrite =
      static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                sys::Memory::MF_WRITE);

  // Compute the total number of pages to allocate.
  size_t TotalSize = 0;
  for (auto &KV : Request) {
    const auto &Seg = KV.second;

    if (Seg.getAlignment() > sys::Process::getPageSizeEstimate())
      return make_error<StringError>("Cannot request higher than page "
                                     "alignment",
                                     inconvertibleErrorCode());

    TotalSize = alignTo(TotalSize, sys::Process::getPageSizeEstimate());
    TotalSize += Seg.getContentSize();
    TotalSize += Seg.getZeroFillSize();
  }

  // Allocate one slab to cover all the segments.
  std::error_code EC;
  auto SlabRemaining =
      sys::Memory::allocateMappedMemory(TotalSize, nullptr, ReadWrite, EC);

  if (EC)
    return errorCodeToError(EC);

  // Allocate segment memory from the slab.
  for (auto &KV : Request) {

    const auto &Seg = KV.second;

    uint64_t SegmentSize = alignTo(Seg.getContentSize() + Seg.getZeroFillSize(),
                                   sys::Process::getPageSizeEstimate());
    assert(SlabRemaining.allocatedSize() >= SegmentSize &&
           "Mapping exceeds allocation");

    sys::MemoryBlock SegMem(SlabRemaining.base(), SegmentSize);
    SlabRemaining = sys::MemoryBlock((char *)SlabRemaining.base() + SegmentSize,
                                     SlabRemaining.allocatedSize() - SegmentSize);

    // Zero out the zero-fill memory.
    memset(static_cast<char *>(SegMem.base()) + Seg.getContentSize(), 0,
           Seg.getZeroFillSize());

    // Record the block for this segment.
    Blocks[KV.first] = std::move(SegMem);
  }

  return std::unique_ptr<InProcessMemoryManager::Allocation>(
      new IPMMAlloc(std::move(Blocks)));
}

} // end namespace jitlink
} // end namespace llvm
