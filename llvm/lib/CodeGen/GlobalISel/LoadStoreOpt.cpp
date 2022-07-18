//===- LoadStoreOpt.cpp ----------- Generic memory optimizations -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the LoadStoreOpt optimization pass.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/LoadStoreOpt.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

#define DEBUG_TYPE "loadstore-opt"

using namespace llvm;
using namespace ore;
using namespace MIPatternMatch;

STATISTIC(NumStoresMerged, "Number of stores merged");

const unsigned MaxStoreSizeToForm = 128;

char LoadStoreOpt::ID = 0;
INITIALIZE_PASS_BEGIN(LoadStoreOpt, DEBUG_TYPE, "Generic memory optimizations",
                      false, false)
INITIALIZE_PASS_END(LoadStoreOpt, DEBUG_TYPE, "Generic memory optimizations",
                    false, false)

LoadStoreOpt::LoadStoreOpt(std::function<bool(const MachineFunction &)> F)
    : MachineFunctionPass(ID), DoNotRunPass(F) {}

LoadStoreOpt::LoadStoreOpt()
    : LoadStoreOpt([](const MachineFunction &) { return false; }) {}

void LoadStoreOpt::init(MachineFunction &MF) {
  this->MF = &MF;
  MRI = &MF.getRegInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  TLI = MF.getSubtarget().getTargetLowering();
  LI = MF.getSubtarget().getLegalizerInfo();
  Builder.setMF(MF);
  IsPreLegalizer = !MF.getProperties().hasProperty(
      MachineFunctionProperties::Property::Legalized);
  InstsToErase.clear();
}

void LoadStoreOpt::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AAResultsWrapperPass>();
  AU.setPreservesAll();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

BaseIndexOffset GISelAddressing::getPointerInfo(Register Ptr,
                                                MachineRegisterInfo &MRI) {
  BaseIndexOffset Info;
  Register PtrAddRHS;
  if (!mi_match(Ptr, MRI, m_GPtrAdd(m_Reg(Info.BaseReg), m_Reg(PtrAddRHS)))) {
    Info.BaseReg = Ptr;
    Info.IndexReg = Register();
    Info.IsIndexSignExt = false;
    return Info;
  }

  auto RHSCst = getIConstantVRegValWithLookThrough(PtrAddRHS, MRI);
  if (RHSCst)
    Info.Offset = RHSCst->Value.getSExtValue();

  // Just recognize a simple case for now. In future we'll need to match
  // indexing patterns for base + index + constant.
  Info.IndexReg = PtrAddRHS;
  Info.IsIndexSignExt = false;
  return Info;
}

bool GISelAddressing::aliasIsKnownForLoadStore(const MachineInstr &MI1,
                                               const MachineInstr &MI2,
                                               bool &IsAlias,
                                               MachineRegisterInfo &MRI) {
  auto *LdSt1 = dyn_cast<GLoadStore>(&MI1);
  auto *LdSt2 = dyn_cast<GLoadStore>(&MI2);
  if (!LdSt1 || !LdSt2)
    return false;

  BaseIndexOffset BasePtr0 = getPointerInfo(LdSt1->getPointerReg(), MRI);
  BaseIndexOffset BasePtr1 = getPointerInfo(LdSt2->getPointerReg(), MRI);

  if (!BasePtr0.BaseReg.isValid() || !BasePtr1.BaseReg.isValid())
    return false;

  int64_t Size1 = LdSt1->getMemSize();
  int64_t Size2 = LdSt2->getMemSize();

  int64_t PtrDiff;
  if (BasePtr0.BaseReg == BasePtr1.BaseReg) {
    PtrDiff = BasePtr1.Offset - BasePtr0.Offset;
    // If the size of memory access is unknown, do not use it to do analysis.
    // One example of unknown size memory access is to load/store scalable
    // vector objects on the stack.
    // BasePtr1 is PtrDiff away from BasePtr0. They alias if none of the
    // following situations arise:
    if (PtrDiff >= 0 &&
        Size1 != static_cast<int64_t>(MemoryLocation::UnknownSize)) {
      // [----BasePtr0----]
      //                         [---BasePtr1--]
      // ========PtrDiff========>
      IsAlias = !(Size1 <= PtrDiff);
      return true;
    }
    if (PtrDiff < 0 &&
        Size2 != static_cast<int64_t>(MemoryLocation::UnknownSize)) {
      //                     [----BasePtr0----]
      // [---BasePtr1--]
      // =====(-PtrDiff)====>
      IsAlias = !((PtrDiff + Size2) <= 0);
      return true;
    }
    return false;
  }

  // If both BasePtr0 and BasePtr1 are FrameIndexes, we will not be
  // able to calculate their relative offset if at least one arises
  // from an alloca. However, these allocas cannot overlap and we
  // can infer there is no alias.
  auto *Base0Def = getDefIgnoringCopies(BasePtr0.BaseReg, MRI);
  auto *Base1Def = getDefIgnoringCopies(BasePtr1.BaseReg, MRI);
  if (!Base0Def || !Base1Def)
    return false; // Couldn't tell anything.


  if (Base0Def->getOpcode() != Base1Def->getOpcode())
    return false;

  if (Base0Def->getOpcode() == TargetOpcode::G_FRAME_INDEX) {
    MachineFrameInfo &MFI = Base0Def->getMF()->getFrameInfo();
    // If the bases have the same frame index but we couldn't find a
    // constant offset, (indices are different) be conservative.
    if (Base0Def != Base1Def &&
        (!MFI.isFixedObjectIndex(Base0Def->getOperand(1).getIndex()) ||
         !MFI.isFixedObjectIndex(Base1Def->getOperand(1).getIndex()))) {
      IsAlias = false;
      return true;
    }
  }

  // This implementation is a lot more primitive than the SDAG one for now.
  // FIXME: what about constant pools?
  if (Base0Def->getOpcode() == TargetOpcode::G_GLOBAL_VALUE) {
    auto GV0 = Base0Def->getOperand(1).getGlobal();
    auto GV1 = Base1Def->getOperand(1).getGlobal();
    if (GV0 != GV1) {
      IsAlias = false;
      return true;
    }
  }

  // Can't tell anything about aliasing.
  return false;
}

bool GISelAddressing::instMayAlias(const MachineInstr &MI,
                                   const MachineInstr &Other,
                                   MachineRegisterInfo &MRI,
                                   AliasAnalysis *AA) {
  struct MemUseCharacteristics {
    bool IsVolatile;
    bool IsAtomic;
    Register BasePtr;
    int64_t Offset;
    uint64_t NumBytes;
    MachineMemOperand *MMO;
  };

  auto getCharacteristics =
      [&](const MachineInstr *MI) -> MemUseCharacteristics {
    if (const auto *LS = dyn_cast<GLoadStore>(MI)) {
      Register BaseReg;
      int64_t Offset = 0;
      // No pre/post-inc addressing modes are considered here, unlike in SDAG.
      if (!mi_match(LS->getPointerReg(), MRI,
                    m_GPtrAdd(m_Reg(BaseReg), m_ICst(Offset)))) {
        BaseReg = LS->getPointerReg();
        Offset = 0;
      }

      uint64_t Size = MemoryLocation::getSizeOrUnknown(
          LS->getMMO().getMemoryType().getSizeInBytes());
      return {LS->isVolatile(),       LS->isAtomic(),          BaseReg,
              Offset /*base offset*/, Size, &LS->getMMO()};
    }
    // FIXME: support recognizing lifetime instructions.
    // Default.
    return {false /*isvolatile*/,
            /*isAtomic*/ false,          Register(),
            (int64_t)0 /*offset*/,       0 /*size*/,
            (MachineMemOperand *)nullptr};
  };
  MemUseCharacteristics MUC0 = getCharacteristics(&MI),
                        MUC1 = getCharacteristics(&Other);

  // If they are to the same address, then they must be aliases.
  if (MUC0.BasePtr.isValid() && MUC0.BasePtr == MUC1.BasePtr &&
      MUC0.Offset == MUC1.Offset)
    return true;

  // If they are both volatile then they cannot be reordered.
  if (MUC0.IsVolatile && MUC1.IsVolatile)
    return true;

  // Be conservative about atomics for the moment
  // TODO: This is way overconservative for unordered atomics (see D66309)
  if (MUC0.IsAtomic && MUC1.IsAtomic)
    return true;

  // If one operation reads from invariant memory, and the other may store, they
  // cannot alias.
  if (MUC0.MMO && MUC1.MMO) {
    if ((MUC0.MMO->isInvariant() && MUC1.MMO->isStore()) ||
        (MUC1.MMO->isInvariant() && MUC0.MMO->isStore()))
      return false;
  }

  // Try to prove that there is aliasing, or that there is no aliasing. Either
  // way, we can return now. If nothing can be proved, proceed with more tests.
  bool IsAlias;
  if (GISelAddressing::aliasIsKnownForLoadStore(MI, Other, IsAlias, MRI))
    return IsAlias;

  // The following all rely on MMO0 and MMO1 being valid.
  if (!MUC0.MMO || !MUC1.MMO)
    return true;

  // FIXME: port the alignment based alias analysis from SDAG's isAlias().
  int64_t SrcValOffset0 = MUC0.MMO->getOffset();
  int64_t SrcValOffset1 = MUC1.MMO->getOffset();
  uint64_t Size0 = MUC0.NumBytes;
  uint64_t Size1 = MUC1.NumBytes;
  if (AA && MUC0.MMO->getValue() && MUC1.MMO->getValue() &&
      Size0 != MemoryLocation::UnknownSize &&
      Size1 != MemoryLocation::UnknownSize) {
    // Use alias analysis information.
    int64_t MinOffset = std::min(SrcValOffset0, SrcValOffset1);
    int64_t Overlap0 = Size0 + SrcValOffset0 - MinOffset;
    int64_t Overlap1 = Size1 + SrcValOffset1 - MinOffset;
    if (AA->isNoAlias(MemoryLocation(MUC0.MMO->getValue(), Overlap0,
                                     MUC0.MMO->getAAInfo()),
                      MemoryLocation(MUC1.MMO->getValue(), Overlap1,
                                     MUC1.MMO->getAAInfo())))
      return false;
  }

  // Otherwise we have to assume they alias.
  return true;
}

/// Returns true if the instruction creates an unavoidable hazard that
/// forces a boundary between store merge candidates.
static bool isInstHardMergeHazard(MachineInstr &MI) {
  return MI.hasUnmodeledSideEffects() || MI.hasOrderedMemoryRef();
}

bool LoadStoreOpt::mergeStores(SmallVectorImpl<GStore *> &StoresToMerge) {
  // Try to merge all the stores in the vector, splitting into separate segments
  // as necessary.
  assert(StoresToMerge.size() > 1 && "Expected multiple stores to merge");
  LLT OrigTy = MRI->getType(StoresToMerge[0]->getValueReg());
  LLT PtrTy = MRI->getType(StoresToMerge[0]->getPointerReg());
  unsigned AS = PtrTy.getAddressSpace();
  // Ensure the legal store info is computed for this address space.
  initializeStoreMergeTargetInfo(AS);
  const auto &LegalSizes = LegalStoreSizes[AS];

#ifndef NDEBUG
  for (auto *StoreMI : StoresToMerge)
    assert(MRI->getType(StoreMI->getValueReg()) == OrigTy);
#endif

  const auto &DL = MF->getFunction().getParent()->getDataLayout();
  bool AnyMerged = false;
  do {
    unsigned NumPow2 = PowerOf2Floor(StoresToMerge.size());
    unsigned MaxSizeBits = NumPow2 * OrigTy.getSizeInBits().getFixedSize();
    // Compute the biggest store we can generate to handle the number of stores.
    unsigned MergeSizeBits;
    for (MergeSizeBits = MaxSizeBits; MergeSizeBits > 1; MergeSizeBits /= 2) {
      LLT StoreTy = LLT::scalar(MergeSizeBits);
      EVT StoreEVT =
          getApproximateEVTForLLT(StoreTy, DL, MF->getFunction().getContext());
      if (LegalSizes.size() > MergeSizeBits && LegalSizes[MergeSizeBits] &&
          TLI->canMergeStoresTo(AS, StoreEVT, *MF) &&
          (TLI->isTypeLegal(StoreEVT)))
        break; // We can generate a MergeSize bits store.
    }
    if (MergeSizeBits <= OrigTy.getSizeInBits())
      return AnyMerged; // No greater merge.

    unsigned NumStoresToMerge = MergeSizeBits / OrigTy.getSizeInBits();
    // Perform the actual merging.
    SmallVector<GStore *, 8> SingleMergeStores(
        StoresToMerge.begin(), StoresToMerge.begin() + NumStoresToMerge);
    AnyMerged |= doSingleStoreMerge(SingleMergeStores);
    StoresToMerge.erase(StoresToMerge.begin(),
                        StoresToMerge.begin() + NumStoresToMerge);
  } while (StoresToMerge.size() > 1);
  return AnyMerged;
}

bool LoadStoreOpt::isLegalOrBeforeLegalizer(const LegalityQuery &Query,
                                            MachineFunction &MF) const {
  auto Action = LI->getAction(Query).Action;
  // If the instruction is unsupported, it can't be legalized at all.
  if (Action == LegalizeActions::Unsupported)
    return false;
  return IsPreLegalizer || Action == LegalizeAction::Legal;
}

bool LoadStoreOpt::doSingleStoreMerge(SmallVectorImpl<GStore *> &Stores) {
  assert(Stores.size() > 1);
  // We know that all the stores are consecutive and there are no aliasing
  // operations in the range. However, the values that are being stored may be
  // generated anywhere before each store. To ensure we have the values
  // available, we materialize the wide value and new store at the place of the
  // final store in the merge sequence.
  GStore *FirstStore = Stores[0];
  const unsigned NumStores = Stores.size();
  LLT SmallTy = MRI->getType(FirstStore->getValueReg());
  LLT WideValueTy =
      LLT::scalar(NumStores * SmallTy.getSizeInBits().getFixedSize());

  // For each store, compute pairwise merged debug locs.
  DebugLoc MergedLoc;
  for (unsigned AIdx = 0, BIdx = 1; BIdx < NumStores; ++AIdx, ++BIdx)
    MergedLoc = DILocation::getMergedLocation(Stores[AIdx]->getDebugLoc(),
                                              Stores[BIdx]->getDebugLoc());
  Builder.setInstr(*Stores.back());
  Builder.setDebugLoc(MergedLoc);

  // If all of the store values are constants, then create a wide constant
  // directly. Otherwise, we need to generate some instructions to merge the
  // existing values together into a wider type.
  SmallVector<APInt, 8> ConstantVals;
  for (auto *Store : Stores) {
    auto MaybeCst =
        getIConstantVRegValWithLookThrough(Store->getValueReg(), *MRI);
    if (!MaybeCst) {
      ConstantVals.clear();
      break;
    }
    ConstantVals.emplace_back(MaybeCst->Value);
  }

  Register WideReg;
  auto *WideMMO =
      MF->getMachineMemOperand(&FirstStore->getMMO(), 0, WideValueTy);
  if (ConstantVals.empty()) {
    // Mimic the SDAG behaviour here and don't try to do anything for unknown
    // values. In future, we should also support the cases of loads and
    // extracted vector elements.
    return false;
  }

  assert(ConstantVals.size() == NumStores);
  // Check if our wide constant is legal.
  if (!isLegalOrBeforeLegalizer({TargetOpcode::G_CONSTANT, {WideValueTy}}, *MF))
    return false;
  APInt WideConst(WideValueTy.getSizeInBits(), 0);
  for (unsigned Idx = 0; Idx < ConstantVals.size(); ++Idx) {
    // Insert the smaller constant into the corresponding position in the
    // wider one.
    WideConst.insertBits(ConstantVals[Idx], Idx * SmallTy.getSizeInBits());
  }
  WideReg = Builder.buildConstant(WideValueTy, WideConst).getReg(0);
  auto NewStore =
      Builder.buildStore(WideReg, FirstStore->getPointerReg(), *WideMMO);
  (void) NewStore;
  LLVM_DEBUG(dbgs() << "Created merged store: " << *NewStore);
  NumStoresMerged += Stores.size();

  MachineOptimizationRemarkEmitter MORE(*MF, nullptr);
  MORE.emit([&]() {
    MachineOptimizationRemark R(DEBUG_TYPE, "MergedStore",
                                FirstStore->getDebugLoc(),
                                FirstStore->getParent());
    R << "Merged " << NV("NumMerged", Stores.size()) << " stores of "
      << NV("OrigWidth", SmallTy.getSizeInBytes())
      << " bytes into a single store of "
      << NV("NewWidth", WideValueTy.getSizeInBytes()) << " bytes";
    return R;
  });

  for (auto *MI : Stores)
    InstsToErase.insert(MI);
  return true;
}

bool LoadStoreOpt::processMergeCandidate(StoreMergeCandidate &C) {
  if (C.Stores.size() < 2) {
    C.reset();
    return false;
  }

  LLVM_DEBUG(dbgs() << "Checking store merge candidate with " << C.Stores.size()
                    << " stores, starting with " << *C.Stores[0]);
  // We know that the stores in the candidate are adjacent.
  // Now we need to check if any potential aliasing instructions recorded
  // during the search alias with load/stores added to the candidate after.
  // For example, if we have the candidate:
  //   C.Stores = [ST1, ST2, ST3, ST4]
  // and after seeing ST2 we saw a load LD1, which did not alias with ST1 or
  // ST2, then we would have recorded it into the PotentialAliases structure
  // with the associated index value of "1". Then we see ST3 and ST4 and add
  // them to the candidate group. We know that LD1 does not alias with ST1 or
  // ST2, since we already did that check. However we don't yet know if it
  // may alias ST3 and ST4, so we perform those checks now.
  SmallVector<GStore *> StoresToMerge;

  auto DoesStoreAliasWithPotential = [&](unsigned Idx, GStore &CheckStore) {
    for (auto AliasInfo : reverse(C.PotentialAliases)) {
      MachineInstr *PotentialAliasOp = AliasInfo.first;
      unsigned PreCheckedIdx = AliasInfo.second;
      if (static_cast<unsigned>(Idx) > PreCheckedIdx) {
        // Need to check this alias.
        if (GISelAddressing::instMayAlias(CheckStore, *PotentialAliasOp, *MRI,
                                          AA)) {
          LLVM_DEBUG(dbgs() << "Potential alias " << *PotentialAliasOp
                            << " detected\n");
          return true;
        }
      } else {
        // Once our store index is lower than the index associated with the
        // potential alias, we know that we've already checked for this alias
        // and all of the earlier potential aliases too.
        return false;
      }
    }
    return false;
  };
  // Start from the last store in the group, and check if it aliases with any
  // of the potential aliasing operations in the list.
  for (int StoreIdx = C.Stores.size() - 1; StoreIdx >= 0; --StoreIdx) {
    auto *CheckStore = C.Stores[StoreIdx];
    if (DoesStoreAliasWithPotential(StoreIdx, *CheckStore))
      continue;
    StoresToMerge.emplace_back(CheckStore);
  }

  LLVM_DEBUG(dbgs() << StoresToMerge.size()
                    << " stores remaining after alias checks. Merging...\n");

  // Now we've checked for aliasing hazards, merge any stores left.
  C.reset();
  if (StoresToMerge.size() < 2)
    return false;
  return mergeStores(StoresToMerge);
}

bool LoadStoreOpt::operationAliasesWithCandidate(MachineInstr &MI,
                                                 StoreMergeCandidate &C) {
  if (C.Stores.empty())
    return false;
  return llvm::any_of(C.Stores, [&](MachineInstr *OtherMI) {
    return instMayAlias(MI, *OtherMI, *MRI, AA);
  });
}

void LoadStoreOpt::StoreMergeCandidate::addPotentialAlias(MachineInstr &MI) {
  PotentialAliases.emplace_back(std::make_pair(&MI, Stores.size() - 1));
}

bool LoadStoreOpt::addStoreToCandidate(GStore &StoreMI,
                                       StoreMergeCandidate &C) {
  // Check if the given store writes to an adjacent address, and other
  // requirements.
  LLT ValueTy = MRI->getType(StoreMI.getValueReg());
  LLT PtrTy = MRI->getType(StoreMI.getPointerReg());

  // Only handle scalars.
  if (!ValueTy.isScalar())
    return false;

  // Don't allow truncating stores for now.
  if (StoreMI.getMemSizeInBits() != ValueTy.getSizeInBits())
    return false;

  // Avoid adding volatile or ordered stores to the candidate. We already have a
  // check for this in instMayAlias() but that only get's called later between
  // potential aliasing hazards.
  if (!StoreMI.isSimple())
    return false;

  Register StoreAddr = StoreMI.getPointerReg();
  auto BIO = getPointerInfo(StoreAddr, *MRI);
  Register StoreBase = BIO.BaseReg;
  uint64_t StoreOffCst = BIO.Offset;
  if (C.Stores.empty()) {
    // This is the first store of the candidate.
    // If the offset can't possibly allow for a lower addressed store with the
    // same base, don't bother adding it.
    if (StoreOffCst < ValueTy.getSizeInBytes())
      return false;
    C.BasePtr = StoreBase;
    C.CurrentLowestOffset = StoreOffCst;
    C.Stores.emplace_back(&StoreMI);
    LLVM_DEBUG(dbgs() << "Starting a new merge candidate group with: "
                      << StoreMI);
    return true;
  }

  // Check the store is the same size as the existing ones in the candidate.
  if (MRI->getType(C.Stores[0]->getValueReg()).getSizeInBits() !=
      ValueTy.getSizeInBits())
    return false;

  if (MRI->getType(C.Stores[0]->getPointerReg()).getAddressSpace() !=
      PtrTy.getAddressSpace())
    return false;

  // There are other stores in the candidate. Check that the store address
  // writes to the next lowest adjacent address.
  if (C.BasePtr != StoreBase)
    return false;
  if ((C.CurrentLowestOffset - ValueTy.getSizeInBytes()) !=
      static_cast<uint64_t>(StoreOffCst))
    return false;

  // This writes to an adjacent address. Allow it.
  C.Stores.emplace_back(&StoreMI);
  C.CurrentLowestOffset = C.CurrentLowestOffset - ValueTy.getSizeInBytes();
  LLVM_DEBUG(dbgs() << "Candidate added store: " << StoreMI);
  return true;
}

bool LoadStoreOpt::mergeBlockStores(MachineBasicBlock &MBB) {
  bool Changed = false;
  // Walk through the block bottom-up, looking for merging candidates.
  StoreMergeCandidate Candidate;
  for (MachineInstr &MI : llvm::reverse(MBB)) {
    if (InstsToErase.contains(&MI))
      continue;

    if (auto *StoreMI = dyn_cast<GStore>(&MI)) {
      // We have a G_STORE. Add it to the candidate if it writes to an adjacent
      // address.
      if (!addStoreToCandidate(*StoreMI, Candidate)) {
        // Store wasn't eligible to be added. May need to record it as a
        // potential alias.
        if (operationAliasesWithCandidate(*StoreMI, Candidate)) {
          Changed |= processMergeCandidate(Candidate);
          continue;
        }
        Candidate.addPotentialAlias(*StoreMI);
      }
      continue;
    }

    // If we don't have any stores yet, this instruction can't pose a problem.
    if (Candidate.Stores.empty())
      continue;

    // We're dealing with some other kind of instruction.
    if (isInstHardMergeHazard(MI)) {
      Changed |= processMergeCandidate(Candidate);
      Candidate.Stores.clear();
      continue;
    }

    if (!MI.mayLoadOrStore())
      continue;

    if (operationAliasesWithCandidate(MI, Candidate)) {
      // We have a potential alias, so process the current candidate if we can
      // and then continue looking for a new candidate.
      Changed |= processMergeCandidate(Candidate);
      continue;
    }

    // Record this instruction as a potential alias for future stores that are
    // added to the candidate.
    Candidate.addPotentialAlias(MI);
  }

  // Process any candidate left after finishing searching the entire block.
  Changed |= processMergeCandidate(Candidate);

  // Erase instructions now that we're no longer iterating over the block.
  for (auto *MI : InstsToErase)
    MI->eraseFromParent();
  InstsToErase.clear();
  return Changed;
}

bool LoadStoreOpt::mergeFunctionStores(MachineFunction &MF) {
  bool Changed = false;
  for (auto &BB : MF) {
    Changed |= mergeBlockStores(BB);
  }
  return Changed;
}

void LoadStoreOpt::initializeStoreMergeTargetInfo(unsigned AddrSpace) {
  // Query the legalizer info to record what store types are legal.
  // We record this because we don't want to bother trying to merge stores into
  // illegal ones, which would just result in being split again.

  if (LegalStoreSizes.count(AddrSpace)) {
    assert(LegalStoreSizes[AddrSpace].any());
    return; // Already cached sizes for this address space.
  }

  // Need to reserve at least MaxStoreSizeToForm + 1 bits.
  BitVector LegalSizes(MaxStoreSizeToForm * 2);
  const auto &LI = *MF->getSubtarget().getLegalizerInfo();
  const auto &DL = MF->getFunction().getParent()->getDataLayout();
  Type *IntPtrIRTy =
      DL.getIntPtrType(MF->getFunction().getContext(), AddrSpace);
  LLT PtrTy = getLLTForType(*IntPtrIRTy->getPointerTo(AddrSpace), DL);
  // We assume that we're not going to be generating any stores wider than
  // MaxStoreSizeToForm bits for now.
  for (unsigned Size = 2; Size <= MaxStoreSizeToForm; Size *= 2) {
    LLT Ty = LLT::scalar(Size);
    SmallVector<LegalityQuery::MemDesc, 2> MemDescrs(
        {{Ty, Ty.getSizeInBits(), AtomicOrdering::NotAtomic}});
    SmallVector<LLT> StoreTys({Ty, PtrTy});
    LegalityQuery Q(TargetOpcode::G_STORE, StoreTys, MemDescrs);
    LegalizeActionStep ActionStep = LI.getAction(Q);
    if (ActionStep.Action == LegalizeActions::Legal)
      LegalSizes.set(Size);
  }
  assert(LegalSizes.any() && "Expected some store sizes to be legal!");
  LegalStoreSizes[AddrSpace] = LegalSizes;
}

bool LoadStoreOpt::runOnMachineFunction(MachineFunction &MF) {
  // If the ISel pipeline failed, do not bother running that pass.
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  LLVM_DEBUG(dbgs() << "Begin memory optimizations for: " << MF.getName()
                    << '\n');

  init(MF);
  bool Changed = false;
  Changed |= mergeFunctionStores(MF);

  LegalStoreSizes.clear();
  return Changed;
}
