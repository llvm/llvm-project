//===---- EPCGenericJITLinkMemoryManager.cpp -- Mem management via EPC ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/LookupAndRecordAddrs.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"

#include <limits>

namespace llvm {
namespace orc {

class EPCGenericJITLinkMemoryManager::Alloc
    : public jitlink::JITLinkMemoryManager::Allocation {
public:
  struct SegInfo {
    char *WorkingMem = nullptr;
    ExecutorAddr TargetAddr;
    uint64_t ContentSize = 0;
    uint64_t ZeroFillSize = 0;
  };
  using SegInfoMap = DenseMap<unsigned, SegInfo>;

  Alloc(EPCGenericJITLinkMemoryManager &Parent, ExecutorAddr TargetAddr,
        std::unique_ptr<char[]> WorkingBuffer, SegInfoMap Segs)
      : Parent(Parent), TargetAddr(TargetAddr),
        WorkingBuffer(std::move(WorkingBuffer)), Segs(std::move(Segs)) {}

  MutableArrayRef<char> getWorkingMemory(ProtectionFlags Seg) override {
    auto I = Segs.find(Seg);
    assert(I != Segs.end() && "No allocation for seg");
    assert(I->second.ContentSize <= std::numeric_limits<size_t>::max());
    return {I->second.WorkingMem, static_cast<size_t>(I->second.ContentSize)};
  }

  JITTargetAddress getTargetMemory(ProtectionFlags Seg) override {
    auto I = Segs.find(Seg);
    assert(I != Segs.end() && "No allocation for seg");
    return I->second.TargetAddr.getValue();
  }

  void finalizeAsync(FinalizeContinuation OnFinalize) override {
    char *WorkingMem = WorkingBuffer.get();
    tpctypes::FinalizeRequest FR;
    for (auto &KV : Segs) {
      assert(KV.second.ContentSize <= std::numeric_limits<size_t>::max());
      FR.Segments.push_back(tpctypes::SegFinalizeRequest{
          tpctypes::toWireProtectionFlags(
              static_cast<sys::Memory::ProtectionFlags>(KV.first)),
          KV.second.TargetAddr,
          alignTo(KV.second.ContentSize + KV.second.ZeroFillSize,
                  Parent.EPC.getPageSize()),
          {WorkingMem, static_cast<size_t>(KV.second.ContentSize)}});
      WorkingMem += KV.second.ContentSize;
    }
    Parent.EPC.callSPSWrapperAsync<
        rt::SPSSimpleExecutorMemoryManagerFinalizeSignature>(
        Parent.SAs.Finalize,
        [OnFinalize = std::move(OnFinalize)](Error SerializationErr,
                                             Error FinalizeErr) {
          if (SerializationErr)
            OnFinalize(std::move(SerializationErr));
          else
            OnFinalize(std::move(FinalizeErr));
        },
        Parent.SAs.Allocator, std::move(FR));
  }

  Error deallocate() override {
    Error Err = Error::success();
    if (auto E2 = Parent.EPC.callSPSWrapper<
                  rt::SPSSimpleExecutorMemoryManagerDeallocateSignature>(
            Parent.SAs.Deallocate, Err, Parent.SAs.Allocator,
            ArrayRef<ExecutorAddr>(TargetAddr)))
      return E2;
    return Err;
  }

private:
  EPCGenericJITLinkMemoryManager &Parent;
  ExecutorAddr TargetAddr;
  std::unique_ptr<char[]> WorkingBuffer;
  SegInfoMap Segs;
};

Expected<std::unique_ptr<jitlink::JITLinkMemoryManager::Allocation>>
EPCGenericJITLinkMemoryManager::allocate(const jitlink::JITLinkDylib *JD,
                                         const SegmentsRequestMap &Request) {
  Alloc::SegInfoMap Segs;
  uint64_t AllocSize = 0;
  size_t WorkingSize = 0;
  for (auto &KV : Request) {
    if (!isPowerOf2_64(KV.second.getAlignment()))
      return make_error<StringError>("Alignment is not a power of two",
                                     inconvertibleErrorCode());
    if (KV.second.getAlignment() > EPC.getPageSize())
      return make_error<StringError>("Alignment exceeds page size",
                                     inconvertibleErrorCode());

    auto &Seg = Segs[KV.first];
    Seg.ContentSize = KV.second.getContentSize();
    Seg.ZeroFillSize = KV.second.getZeroFillSize();
    AllocSize += alignTo(Seg.ContentSize + Seg.ZeroFillSize, EPC.getPageSize());
    WorkingSize += Seg.ContentSize;
  }

  std::unique_ptr<char[]> WorkingBuffer;
  if (WorkingSize > 0)
    WorkingBuffer = std::make_unique<char[]>(WorkingSize);
  Expected<ExecutorAddr> TargetAllocAddr((ExecutorAddr()));
  if (auto Err = EPC.callSPSWrapper<
                 rt::SPSSimpleExecutorMemoryManagerReserveSignature>(
          SAs.Reserve, TargetAllocAddr, SAs.Allocator, AllocSize))
    return std::move(Err);
  if (!TargetAllocAddr)
    return TargetAllocAddr.takeError();

  char *WorkingMem = WorkingBuffer.get();
  JITTargetAddress SegAddr = TargetAllocAddr->getValue();
  for (auto &KV : Segs) {
    auto &Seg = KV.second;
    Seg.TargetAddr.setValue(SegAddr);
    SegAddr += alignTo(Seg.ContentSize + Seg.ZeroFillSize, EPC.getPageSize());
    Seg.WorkingMem = WorkingMem;
    WorkingMem += Seg.ContentSize;
  }

  return std::make_unique<Alloc>(*this, *TargetAllocAddr,
                                 std::move(WorkingBuffer), std::move(Segs));
}

} // end namespace orc
} // end namespace llvm
