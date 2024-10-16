//===- SeedCollector.cpp  -0000000-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SeedCollector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Type.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
namespace llvm::sandboxir {

cl::opt<unsigned> SeedBundleSizeLimit(
    "sbvec-seed-bundle-size-limit", cl::init(32), cl::Hidden,
    cl::desc("Limit the size of the seed bundle to cap compilation time."));

MutableArrayRef<Instruction *> SeedBundle::getSlice(unsigned StartIdx,
                                                    unsigned MaxVecRegBits,
                                                    bool ForcePowerOf2) {
  // Use uint32_t here for compatibility with IsPowerOf2_32

  // BitCount tracks the size of the working slice. From that we can tell
  // when the working slice's size is a power-of-two and when it exceeds
  // the legal size in MaxVecBits.
  uint32_t BitCount = 0;
  uint32_t NumElements = 0;
  // Tracks the most recent slice where NumElements gave a power-of-2 BitCount
  uint32_t NumElementsPowerOfTwo = 0;
  uint32_t BitCountPowerOfTwo = 0;
  // Can't start a slice with a used instruction.
  assert(!isUsed(StartIdx) && "Expected unused at StartIdx");
  for (auto S : make_range(Seeds.begin() + StartIdx, Seeds.end())) {
    uint32_t InstBits = Utils::getNumBits(S);
    // Stop if this instruction is used, or if adding it puts the slice over
    // the limit.
    if (isUsed(StartIdx + NumElements) || BitCount + InstBits > MaxVecRegBits)
      break;
    NumElements++;
    BitCount += InstBits;
    if (ForcePowerOf2 && isPowerOf2_32(BitCount)) {
      NumElementsPowerOfTwo = NumElements;
      BitCountPowerOfTwo = BitCount;
    }
  }
  if (ForcePowerOf2) {
    NumElements = NumElementsPowerOfTwo;
    BitCount = BitCountPowerOfTwo;
  }

  assert((!ForcePowerOf2 || isPowerOf2_32(BitCount)) &&
         "Must be a power of two");
  // Return any non-empty slice
  if (NumElements > 1)
    return MutableArrayRef<Instruction *>(&Seeds[StartIdx], NumElements);
  else
    return {};
}

template <typename LoadOrStoreT>
SeedContainer::KeyT SeedContainer::getKey(LoadOrStoreT *LSI) const {
  assert((isa<LoadInst>(LSI) || isa<StoreInst>(LSI)) &&
         "Expected Load or Store!");
  Value *Ptr = Utils::getMemInstructionBase(LSI);
  Instruction::Opcode Op = LSI->getOpcode();
  Type *Ty = Utils::getExpectedType(LSI);
  if (auto *VTy = dyn_cast<VectorType>(Ty))
    Ty = VTy->getElementType();
  return {Ptr, Ty, Op};
}

// Explicit instantiations
template SeedContainer::KeyT
SeedContainer::getKey<LoadInst>(LoadInst *LSI) const;
template SeedContainer::KeyT
SeedContainer::getKey<StoreInst>(StoreInst *LSI) const;

bool SeedContainer::erase(Instruction *I) {
  assert((isa<LoadInst>(I) || isa<StoreInst>(I)) && "Expected Load or Store!");
  auto It = SeedLookupMap.find(I);
  if (It == SeedLookupMap.end())
    return false;
  SeedBundle *Bndl = It->second;
  Bndl->setUsed(I);
  return true;
}

template <typename LoadOrStoreT> void SeedContainer::insert(LoadOrStoreT *LSI) {
  // Find the bundle containing seeds for this symbol and type-of-access.
  auto &BundleVec = Bundles[getKey(LSI)];
  // Fill this vector of bundles front to back so that only the last bundle in
  // the vector may have available space. This avoids iteration to find one with
  // space.
  if (BundleVec.empty() || BundleVec.back()->size() == SeedBundleSizeLimit)
    BundleVec.emplace_back(std::make_unique<MemSeedBundle<LoadOrStoreT>>(LSI));
  else
    BundleVec.back()->insert(LSI, SE);

  SeedLookupMap[LSI] = BundleVec.back().get();
}

// Explicit instantiations
template void SeedContainer::insert<LoadInst>(LoadInst *);
template void SeedContainer::insert<StoreInst>(StoreInst *);

#ifndef NDEBUG
void SeedContainer::dump() const {
  for (const auto &Pair : Bundles) {
    auto [I, Ty, Opc] = Pair.first;
    const auto &SeedsVec = Pair.second;
    std::string RefType = dyn_cast<LoadInst>(I)    ? "Load"
                          : dyn_cast<StoreInst>(I) ? "Store"
                                                   : "Other";
    dbgs() << "[Inst=" << *I << " Ty=" << Ty << " " << RefType << "]\n";
    for (const auto &SeedPtr : SeedsVec) {
      SeedPtr->dump(dbgs());
      dbgs() << "\n";
    }
  }
  dbgs() << "\n";
}
#endif // NDEBUG

} // namespace llvm::sandboxir
