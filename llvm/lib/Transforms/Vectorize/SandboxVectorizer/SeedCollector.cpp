//===- SeedCollector.cpp  -0000000-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SeedCollector.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Type.h"
#include "llvm/SandboxIR/Instruction.h"
#include "llvm/SandboxIR/Utils.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
namespace llvm::sandboxir {

cl::opt<unsigned> SeedBundleSizeLimit(
    "sbvec-seed-bundle-size-limit", cl::init(32), cl::Hidden,
    cl::desc("Limit the size of the seed bundle to cap compilation time."));
#define LoadSeedsDef "loads"
#define StoreSeedsDef "stores"
cl::opt<std::string> CollectSeeds(
    "sbvec-collect-seeds", cl::init(LoadSeedsDef "," StoreSeedsDef), cl::Hidden,
    cl::desc("Collect these seeds. Use empty for none or a comma-separated "
             "list of '" LoadSeedsDef "' and '" StoreSeedsDef "'."));
cl::opt<unsigned> SeedGroupsLimit(
    "sbvec-seed-groups-limit", cl::init(256), cl::Hidden,
    cl::desc("Limit the number of collected seeds groups in a BB to "
             "cap compilation time."));

ArrayRef<Instruction *> SeedBundle::getSlice(unsigned StartIdx,
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
    // Stop if this instruction is used. This needs to be done before
    // getNumBits() because a "used" instruction may have been erased.
    if (isUsed(StartIdx + NumElements))
      break;
    uint32_t InstBits = Utils::getNumBits(S);
    // Stop if adding it puts the slice over the limit.
    if (BitCount + InstBits > MaxVecRegBits)
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
    return ArrayRef<Instruction *>(&Seeds[StartIdx], NumElements);
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
void SeedContainer::print(raw_ostream &OS) const {
  for (const auto &Pair : Bundles) {
    auto [I, Ty, Opc] = Pair.first;
    const auto &SeedsVec = Pair.second;
    std::string RefType = dyn_cast<LoadInst>(I)    ? "Load"
                          : dyn_cast<StoreInst>(I) ? "Store"
                                                   : "Other";
    OS << "[Inst=" << *I << " Ty=" << Ty << " " << RefType << "]\n";
    for (const auto &SeedPtr : SeedsVec) {
      SeedPtr->dump(OS);
      OS << "\n";
    }
  }
  OS << "\n";
}

LLVM_DUMP_METHOD void SeedContainer::dump() const { print(dbgs()); }
#endif // NDEBUG

template <typename LoadOrStoreT> static bool isValidMemSeed(LoadOrStoreT *LSI) {
  if (!LSI->isSimple())
    return false;
  auto *Ty = Utils::getExpectedType(LSI);
  // Omit types that are architecturally unvectorizable
  if (Ty->isX86_FP80Ty() || Ty->isPPC_FP128Ty())
    return false;
  // Omit vector types without compile-time-known lane counts
  if (isa<ScalableVectorType>(Ty))
    return false;
  if (auto *VTy = dyn_cast<FixedVectorType>(Ty))
    return VectorType::isValidElementType(VTy->getElementType());
  return VectorType::isValidElementType(Ty);
}

template bool isValidMemSeed<LoadInst>(LoadInst *LSI);
template bool isValidMemSeed<StoreInst>(StoreInst *LSI);

SeedCollector::SeedCollector(BasicBlock *BB, ScalarEvolution &SE)
    : StoreSeeds(SE), LoadSeeds(SE), Ctx(BB->getContext()) {

  bool CollectStores = CollectSeeds.find(StoreSeedsDef) != std::string::npos;
  bool CollectLoads = CollectSeeds.find(LoadSeedsDef) != std::string::npos;
  if (!CollectStores && !CollectLoads)
    return;

  EraseCallbackID = Ctx.registerEraseInstrCallback([this](Instruction *I) {
    if (auto SI = dyn_cast<StoreInst>(I))
      StoreSeeds.erase(SI);
    else if (auto LI = dyn_cast<LoadInst>(I))
      LoadSeeds.erase(LI);
  });

  // Actually collect the seeds.
  for (auto &I : *BB) {
    if (StoreInst *SI = dyn_cast<StoreInst>(&I))
      if (CollectStores && isValidMemSeed(SI))
        StoreSeeds.insert(SI);
    if (LoadInst *LI = dyn_cast<LoadInst>(&I))
      if (CollectLoads && isValidMemSeed(LI))
        LoadSeeds.insert(LI);
    // Cap compilation time.
    if (totalNumSeedGroups() > SeedGroupsLimit)
      break;
  }
}

SeedCollector::~SeedCollector() {
  Ctx.unregisterEraseInstrCallback(EraseCallbackID);
}

#ifndef NDEBUG
void SeedCollector::print(raw_ostream &OS) const {
  OS << "=== StoreSeeds ===\n";
  StoreSeeds.print(OS);
  OS << "=== LoadSeeds ===\n";
  LoadSeeds.print(OS);
}

void SeedCollector::dump() const { print(dbgs()); }
#endif

} // namespace llvm::sandboxir
