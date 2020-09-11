#include "ThinLtoInstrumentationLayer.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"

#include <cstdlib>

#define DEBUG_TYPE "thinltojit"

namespace llvm {
namespace orc {

// TODO: Fixed set of flags may not always be enough. Make this expandable.
void ThinLtoInstrumentationLayer::allocateDiscoveryFlags(unsigned MinFlags) {
  // Round up to full memory pages.
  unsigned PageSize = sys::Process::getPageSizeEstimate();
  unsigned NumPagesEach = (MinFlags + (PageSize - 1)) / PageSize;
  unsigned NumPagesTotal = 2 * NumPagesEach;
  assert(isPowerOf2_64(PageSize) && "Adjust aligned memory alloc below");

  // Allocate one more page to make up for size loss due to alignment.
  void *Storage = std::calloc(NumPagesTotal + 1, PageSize);
  uint64_t StorageAddr = reinterpret_cast<uint64_t>(Storage);
  uint64_t PageSizeDecr = PageSize - 1;
  uint64_t AlignedAddr = ((StorageAddr + PageSizeDecr) & ~PageSizeDecr);
  uint64_t Diff = AlignedAddr - StorageAddr;

  // For each flag we allocate one byte in each location: Incoming and Handled.
  // TODO: 'Handled' could be a bitset, but size must be dynamic
  NumFlagsUsed.store(0);
  NumFlagsAllocated = NumPagesEach * PageSize;
  FlagsStorage = static_cast<uint8_t *>(Storage);
  FlagsIncoming = reinterpret_cast<Flag *>(FlagsStorage + Diff);
  FlagsHandled = FlagsIncoming + NumFlagsAllocated;

  static_assert(sizeof(FlagsIncoming[0]) == sizeof(uint8_t), "Flags are bytes");
  assert(reinterpret_cast<uint64_t>(FlagsIncoming) % PageSize == 0);
  assert(reinterpret_cast<uint64_t>(FlagsHandled) % PageSize == 0);
  assert(NumFlagsAllocated >= MinFlags);
}

// Reserve a new set of discovery flags and return the index of the first one.
unsigned ThinLtoInstrumentationLayer::reserveDiscoveryFlags(unsigned Count) {
#ifndef NDEBUG
  for (unsigned i = NumFlagsUsed.load(), e = i + Count; i < e; i++) {
    assert(FlagsIncoming[i] == Clear);
  }
#endif

  assert(Count > 0);
  return NumFlagsUsed.fetch_add(Count);
}

void ThinLtoInstrumentationLayer::registerDiscoveryFlagOwners(
    std::vector<GlobalValue::GUID> Guids, unsigned FirstIdx) {
  unsigned Count = Guids.size();

  std::lock_guard<std::mutex> Lock(DiscoveryFlagsInfoLock);
  for (unsigned i = 0; i < Count; i++) {
    assert(!FlagOwnersMap.count(FirstIdx + i) &&
           "Flag should not have an owner at this point");
    FlagOwnersMap[FirstIdx + i] = Guids[i];
  }
}

std::vector<unsigned> ThinLtoInstrumentationLayer::takeFlagsThatFired() {
  // This is only effective with the respective Release.
  FlagsSync.load(std::memory_order_acquire);

  std::vector<unsigned> Indexes;
  unsigned NumIndexesUsed = NumFlagsUsed.load();
  for (unsigned i = 0; i < NumIndexesUsed; i++) {
    if (FlagsIncoming[i] == Fired && FlagsHandled[i] == Clear) {
      FlagsHandled[i] = Fired;
      Indexes.push_back(i);
    }
  }

  return Indexes;
}

std::vector<GlobalValue::GUID>
ThinLtoInstrumentationLayer::takeFlagOwners(std::vector<unsigned> Indexes) {
  std::vector<GlobalValue::GUID> ReachedFunctions;
  std::lock_guard<std::mutex> Lock(DiscoveryFlagsInfoLock);

  for (unsigned i : Indexes) {
    auto KV = FlagOwnersMap.find(i);
    assert(KV != FlagOwnersMap.end());
    ReachedFunctions.push_back(KV->second);
    FlagOwnersMap.erase(KV);
  }

  return ReachedFunctions;
}

void ThinLtoInstrumentationLayer::nudgeIntoDiscovery(
    std::vector<GlobalValue::GUID> Functions) {
  unsigned Count = Functions.size();

  // Registering synthetic flags in advance. We expect them to get processed
  // before the respective functions get emitted. If not, the emit() function
  unsigned FirstFlagIdx = reserveDiscoveryFlags(Functions.size());
  registerDiscoveryFlagOwners(std::move(Functions), FirstFlagIdx);

  // Initialize the flags as fired and force a cache sync, so discovery will
  // pick them up as soon as possible.
  for (unsigned i = FirstFlagIdx; i < FirstFlagIdx + Count; i++) {
    FlagsIncoming[i] = Fired;
  }
  if (MemFence & ThinLtoJIT::FenceStaticCode) {
    FlagsSync.store(0, std::memory_order_release);
  }

  LLVM_DEBUG(dbgs() << "Nudged " << Count << " new functions into discovery\n");
}

void ThinLtoInstrumentationLayer::emit(MaterializationResponsibility R,
                                       ThreadSafeModule TSM) {
  TSM.withModuleDo([this](Module &M) {
    std::vector<Function *> FunctionsToInstrument;

    // We may have discovered ahead of some functions already, but we still
    // instrument them all. Their notifications steer the future direction of
    // discovery.
    for (Function &F : M.getFunctionList())
      if (!F.isDeclaration())
        FunctionsToInstrument.push_back(&F);

    if (!FunctionsToInstrument.empty()) {
      IRBuilder<> B(M.getContext());
      std::vector<GlobalValue::GUID> NewDiscoveryRoots;

      // Flags that fire must have owners registered. We will do it below and
      // that's fine, because they can only be reached once the code is emitted.
      unsigned FirstFlagIdx =
          reserveDiscoveryFlags(FunctionsToInstrument.size());

      unsigned NextFlagIdx = FirstFlagIdx;
      for (Function *F : FunctionsToInstrument) {
        // TODO: Emitting the write operation into an indirection stub would
        // allow to skip it once we got the notification.
        BasicBlock *E = &F->getEntryBlock();
        B.SetInsertPoint(BasicBlock::Create(
            M.getContext(), "NotifyFunctionReachedProlog", F, E));
        compileFunctionReachedFlagSetter(B, FlagsIncoming + NextFlagIdx);
        B.CreateBr(E);

        std::string GlobalName = GlobalValue::getGlobalIdentifier(
            F->getName(), F->getLinkage(), M.getSourceFileName());
        NewDiscoveryRoots.push_back(GlobalValue::getGUID(GlobalName));
        ++NextFlagIdx;
      }

      LLVM_DEBUG(dbgs() << "Instrumented " << NewDiscoveryRoots.size()
                        << " new functions in module " << M.getName() << "\n");

      // Submit owner info, so the DiscoveryThread can evaluate the flags.
      registerDiscoveryFlagOwners(std::move(NewDiscoveryRoots), FirstFlagIdx);
    }
  });

  BaseLayer.emit(std::move(R), std::move(TSM));
}

void ThinLtoInstrumentationLayer::compileFunctionReachedFlagSetter(
    IRBuilder<> &B, Flag *F) {
  assert(*F == Clear);
  Type *Int64Ty = Type::getInt64Ty(B.getContext());

  // Write one immediate 8bit value to a fixed location in memory.
  auto FlagAddr = pointerToJITTargetAddress(F);
  Type *FlagTy = Type::getInt8Ty(B.getContext());
  B.CreateStore(ConstantInt::get(FlagTy, Fired),
                B.CreateIntToPtr(ConstantInt::get(Int64Ty, FlagAddr),
                                 FlagTy->getPointerTo()));

  if (MemFence & ThinLtoJIT::FenceJITedCode) {
    // Overwrite the sync value with Release ordering. The discovery thread
    // reads it with Acquire ordering. The actual value doesn't matter.
    static constexpr bool IsVolatile = true;
    static constexpr Instruction *NoInsertBefore = nullptr;
    auto SyncFlagAddr = pointerToJITTargetAddress(&FlagsSync);

    B.Insert(
        new StoreInst(ConstantInt::get(Int64Ty, 0),
                      B.CreateIntToPtr(ConstantInt::get(Int64Ty, SyncFlagAddr),
                                       Int64Ty->getPointerTo()),
                      IsVolatile, Align(64), AtomicOrdering::Release,
                      SyncScope::System, NoInsertBefore));
  }
}

void ThinLtoInstrumentationLayer::dump(raw_ostream &OS) {
  OS << "Discovery flags stats\n";

  unsigned NumFlagsFired = 0;
  for (unsigned i = 0; i < NumFlagsAllocated; i++) {
    if (FlagsIncoming[i] == Fired)
      ++NumFlagsFired;
  }
  OS << "Alloc:  " << format("%6.d", NumFlagsAllocated) << "\n";
  OS << "Issued: " << format("%6.d", NumFlagsUsed.load()) << "\n";
  OS << "Fired:  " << format("%6.d", NumFlagsFired) << "\n";

  unsigned RemainingFlagOwners = 0;
  for (const auto &_ : FlagOwnersMap) {
    ++RemainingFlagOwners;
    (void)_;
  }
  OS << "\nFlagOwnersMap has " << RemainingFlagOwners
     << " remaining entries.\n";
}

ThinLtoInstrumentationLayer::~ThinLtoInstrumentationLayer() {
  std::free(FlagsStorage);
}

} // namespace orc
} // namespace llvm
