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
InProcessMemoryManager::allocate(const SegmentsRequestMap &Request) {

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
      return reinterpret_cast<JITTargetAddress>(SegBlocks[Seg].base());
    }
    void finalizeAsync(FinalizeContinuation OnFinalize) override {
      OnFinalize(applyProtections());
    }
    Error deallocate() override {
      for (auto &KV : SegBlocks)
        if (auto EC = sys::Memory::releaseMappedMemory(KV.second))
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

  AllocationMap Blocks;
  const sys::Memory::ProtectionFlags ReadWrite =
      static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                sys::Memory::MF_WRITE);

  for (auto &KV : Request) {
    auto &Seg = KV.second;

    if (Seg.getContentAlignment() > sys::Process::getPageSizeEstimate())
      return make_error<StringError>("Cannot request higher than page "
                                     "alignment",
                                     inconvertibleErrorCode());

    if (sys::Process::getPageSizeEstimate() % Seg.getContentAlignment() != 0)
      return make_error<StringError>("Page size is not a multiple of "
                                     "alignment",
                                     inconvertibleErrorCode());

    uint64_t ZeroFillStart =
        alignTo(Seg.getContentSize(), Seg.getZeroFillAlignment());
    uint64_t SegmentSize = ZeroFillStart + Seg.getZeroFillSize();

    std::error_code EC;
    auto SegMem =
        sys::Memory::allocateMappedMemory(SegmentSize, nullptr, ReadWrite, EC);

    if (EC)
      return errorCodeToError(EC);

    // Zero out the zero-fill memory.
    memset(static_cast<char *>(SegMem.base()) + ZeroFillStart, 0,
           Seg.getZeroFillSize());

    // Record the block for this segment.
    Blocks[KV.first] = std::move(SegMem);
  }
  return std::unique_ptr<InProcessMemoryManager::Allocation>(
      new IPMMAlloc(std::move(Blocks)));
}

} // end namespace jitlink
} // end namespace llvm
