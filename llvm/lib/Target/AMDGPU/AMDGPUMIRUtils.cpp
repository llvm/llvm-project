//===------- AMDGPUMIRUtils.cpp - Helpers for MIR passes ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Helper functions for MIR passes.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMIRUtils.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define DEBUG_TYPE "xb-mir-util"
using namespace llvm;

namespace llvm {
bool getNonDebugMBBEnd(MachineBasicBlock::reverse_iterator &BBEnd,
                       MachineBasicBlock &MBB) {
  // R.End doesn't point to the boundary instruction.
  // Skip Debug instr.
  while (BBEnd != MBB.rend() && BBEnd->isDebugInstr())
    BBEnd++;
  return BBEnd != MBB.rend();
}
} // namespace llvm

namespace {
bool isLocalSegment(const LiveRange::Segment *Seg, SlotIndexes *Indexes,
                    SmallDenseSet<MachineBasicBlock *, 2> &TouchedMBBSet) {
  MachineInstr *StartMI = Indexes->getInstructionFromIndex(Seg->start);
  MachineInstr *EndMI = Indexes->getInstructionFromIndex(Seg->end);
  // Treat non inst as not local.
  if (!StartMI || !EndMI)
    return false;
  // is local when parent MBB the same.
  bool IsSameMBB = StartMI->getParent() == EndMI->getParent();
  if (!IsSameMBB)
    return false;
  // Collect touched MBB.
  MachineBasicBlock *MBB = StartMI->getParent();
  TouchedMBBSet.insert(MBB);
  return true;
}

bool isLocalLiveRange(const LiveRange *Range, SlotIndexes *Indexes,
                      SmallDenseSet<MachineBasicBlock *, 2> &TouchedMBBSet) {
  for (const LiveRange::Segment &Seg : Range->segments) {
    if (!isLocalSegment(&Seg, Indexes, TouchedMBBSet))
      return false;
  }
  return true;
}

bool isLocalSegment(const LiveRange::Segment *Seg, SlotIndexes *Indexes) {
  MachineInstr *StartMI = Indexes->getInstructionFromIndex(Seg->start);
  MachineInstr *EndMI = Indexes->getInstructionFromIndex(Seg->end);
  // Treat non inst as not local.
  if (!StartMI || !EndMI)
    return false;
  // is local when parent MBB the same.
  return StartMI->getParent() == EndMI->getParent();
}

bool isLocalLiveRange(const LiveRange *Range, SlotIndexes *Indexes) {
  for (const LiveRange::Segment &Seg : Range->segments) {
    if (!isLocalSegment(&Seg, Indexes))
      return false;
  }
  return true;
}

// LoopInfo contains a mapping from basic block to the innermost loop. Find
// the outermost loop in the loop nest that contains BB.
const MachineLoop *getOutermostLoop(const MachineLoopInfo *LI,
                                    const MachineBasicBlock *BB) {
  const MachineLoop *L = LI->getLoopFor(BB);
  if (L) {
    while (const MachineLoop *Parent = L->getParentLoop())
      L = Parent;
  }
  return L;
}

bool loopContainsBoth(const MachineLoopInfo *LI, const MachineBasicBlock *BB1,
                      const MachineBasicBlock *BB2) {
  const MachineLoop *L1 = getOutermostLoop(LI, BB1);
  const MachineLoop *L2 = getOutermostLoop(LI, BB2);
  return L1 != nullptr && L1 == L2;
}

} // namespace

namespace llvm {

bool isSccLiveAt(llvm::MachineBasicBlock *MBB,
                 llvm::MachineBasicBlock::iterator MI) {
  const TargetRegisterInfo *TRI =
      MBB->getParent()->getRegInfo().getTargetRegisterInfo();
  for (auto It = MI; It != MBB->end(); ++It) {
    const MachineInstr &CurMI = *It;
    // Hit use of scc, it is live.
    if (CurMI.readsRegister(AMDGPU::SCC, TRI))
      return true;
    // Hit def of scc first, not live.
    if (CurMI.definesRegister(AMDGPU::SCC, TRI))
      return false;
  }
  // Reach the end of MBB, check live-ins of MBB successors.
  for (const MachineBasicBlock *Succ : MBB->successors()) {
    if (Succ->isLiveIn(AMDGPU::SCC))
      return true;
  }
  return false;
}

//
// This function is useful for when we need to insert a new
// instruction that defines scc in a block and we need to find
// a location that will not smash the existing value.
//
// Starting at `BeforeInst` it will look backwards to try to find
// a place in the block where scc is dead so we can insert our new
// def there. If no location can be found it will save and restore
// scc around BeforeInst. This way BeforeInst can safely be used
// as the new insert location.
//
MachineBasicBlock::iterator findOrCreateInsertionPointForSccDef(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MI,
    const TargetRegisterInfo *TRI, const SIInstrInfo *TII,
    MachineRegisterInfo *MRI, SccDefInsertPointConstraintFlags Constraints) {
  // If SCC is dead at MI when we can use MI as the insert point.
  if (!llvm::isSccLiveAt(MBB, MI)) {
    return MI;
  }

  const bool CheckForExecWrite =
      Constraints & SccDefInsertPointConstraintFlags::NoExecWrite;

  // Get the starting reverse iterator taking care to handle the MBB->end()
  // case.
  MachineBasicBlock::reverse_iterator Start;
  if (MI == MBB->end()) {
    Start = MBB->rbegin();
  } else {
    Start = MI.getReverse();
  }

  // Otherwise, walk backwards through the block looking for a location where
  // SCC is dead.
  for (MachineBasicBlock::reverse_iterator It = Start, End = MBB->rend();
       It != End; ++It) {
    // If the instruction modifies exec then we cannot use it as
    // an insertion point (if that is a constraint from the caller).
    // The check for EXEC works for both wave64 and wave32 because
    // it will also catch Writes to the subregisters (e.g. exec_lo).
    if (CheckForExecWrite && It->modifiesRegister(AMDGPU::EXEC, TRI)) {
      break;
    }

    if (It->modifiesRegister(AMDGPU::SCC, TRI) &&
        !It->readsRegister(AMDGPU::SCC, TRI)) {
      return It->getIterator();
    }
  }

  // If no safe location can be found in the block we can save and restore
  // SCC around MI. There is no way to directly read or Write SCC so we use
  // s_cselect to read the current value of SCC and s_cmp to Write the saved
  // value back to SCC.
  //
  // The generated code will look like this;
  //
  //      S_CSELECT_B32 %SavedSCC, -1, 0  # Save SCC
  //      <----- Newly created safe insert point.
  //      MI
  //      S_CMP_LG_U32 %SavedSCC, 0       # Restore SCC
  //
  Register TmpScc = MRI->createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
  DebugLoc DL = MI->getDebugLoc();
  BuildMI(*MBB, MI, DL, TII->get(AMDGPU::S_CSELECT_B32), TmpScc)
      .addImm(-1)
      .addImm(0);
  BuildMI(*MBB, std::next(MI->getIterator()), DL,
          TII->get(AMDGPU::S_CMP_LG_U32))
      .addReg(TmpScc, RegState::Kill)
      .addImm(0);

  return MI;
}

// In case like float4 v, v.x used and defined in one block, v.y used and define
// in another block, one live interval could touch more than one MBB.
// TouchedMBBSet is used for scheduling where local live interval could cross
// multiple regions, need to calculate livereg for each region inside touched
// MBB.
bool isLocalLiveInterval(const LiveInterval &LI, SlotIndexes *Indexes,
                         SmallDenseSet<MachineBasicBlock *, 2> &TouchedMBBSet) {
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges()) {
      if (!isLocalLiveRange(&S, Indexes, TouchedMBBSet))
        return false;
    }
  }
  return isLocalLiveRange(&LI, Indexes, TouchedMBBSet);
}

bool isLocalLiveInterval(const LiveInterval &LI, SlotIndexes *Indexes) {
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges()) {
      if (!isLocalLiveRange(&S, Indexes))
        return false;
    }
  }
  return isLocalLiveRange(&LI, Indexes);
}

void dumpLiveSet(const LiveSet &LiveSet, const SIRegisterInfo *SIRI) {

  dbgs() << "\n live set: \n";
  for (auto It : LiveSet) {
    int Reg = It.first;
    dbgs() << printReg(Reg, SIRI);
    if (It.second.any()) {
      dbgs() << " mask:" << It.second.getAsInteger();
    }
    dbgs() << "\n";
  }
}

LaneBitmask getRegMask(const MachineOperand &MO,
                       const MachineRegisterInfo &MRI) {
  // We don't rely on read-undef_ flag because in case of tentative schedule
  // tracking it isn't set correctly yet. This works correctly however since
  // use mask has been tracked before using LIS.
  return MO.getSubReg() == 0
             ? MRI.getMaxLaneMaskForVReg(MO.getReg())
             : MRI.getTargetRegisterInfo()->getSubRegIndexLaneMask(
                   MO.getSubReg());
}

struct Piece {
  unsigned Reg;
  unsigned Offset;
  unsigned Size;
  static SmallVector<Piece, 8> split(std::bitset<32> Mask) {

    SmallVector<Piece, 8> Pieces;
    Piece Piece = {0, 0, 0};
    for (unsigned i = 0; i < 32; i++) {
      if (Mask.test(i)) {
        if (Piece.Size == 0)
          Piece.Offset = i;

        Piece.Size++;
        // Make sure no piece bigger than 8.
        if (Piece.Size == 8) {
          Pieces.emplace_back(Piece);
          Piece.Size = 0;
        }
      } else {
        if (Piece.Size == 0) {
          continue;
        }
        Pieces.emplace_back(Piece);
        Piece.Size = 0;
      }
    }
    return Pieces;
  }
};

static unsigned getNumLanesIn32BitReg(Register Reg, const SIRegisterInfo *SIRI,
                                      const MachineRegisterInfo &MRI) {
  const TargetRegisterClass *RC = SIRI->getRegClassForReg(MRI, Reg);
  const TargetRegisterClass *SubregRC =
      SIRI->getSubRegisterClass(RC, AMDGPU::sub0);
  return SubregRC->LaneMask.getNumLanes();
}

static std::vector<unsigned>
getMinimalSpanningSubRegIdxSetForLaneMask(const TargetRegisterInfo *TRI,
                                          const TargetRegisterClass *RC,
                                          LaneBitmask Mask) {
  // TODO: this could replace the code it was copied from in SplitKit.cpp

  // First pass: Try to find a perfectly matching subregister index.
  // If none exists find the one covering the most lanemask bits.
  SmallVector<unsigned, 8> PossibleIndexes;
  unsigned BestIdx = 0;
  const LaneBitmask Avoid = ~Mask;
  {
    unsigned BestCover = 0;
    for (unsigned Idx = 1, E = TRI->getNumSubRegIndices(); Idx < E; ++Idx) {
      // Is this index even compatible with the given class?
      if (TRI->getSubClassWithSubReg(RC, Idx) != RC)
        continue;
      LaneBitmask SubRegMask = TRI->getSubRegIndexLaneMask(Idx);
      // Early exit if we found a perfect match.
      if (SubRegMask == Mask) {
        BestIdx = Idx;
        break;
      }

      // The index must not cover any lanes outside
      if ((SubRegMask & Avoid).any())
        continue;

      unsigned PopCount = SubRegMask.getNumLanes();
      PossibleIndexes.push_back(Idx);
      if (PopCount > BestCover) {
        BestCover = PopCount;
        BestIdx = Idx;
      }
    }
  }

  // Abort if we cannot possibly implement the COPY with the given indexes.
  if (BestIdx == 0) {
    LLVM_DEBUG(dbgs() << "Unable to find minimal spanning sub register(s) for "
                      << TRI->getRegClassName(RC) << " mask "
                      << PrintLaneMask(Mask) << '\n');
    assert(false && "Impossible to span reg class");
    return std::vector<unsigned>();
  }

  std::vector<unsigned> Result;
  Result.push_back(BestIdx);

  // Greedy heuristic: Keep iterating keeping the best covering subreg index
  // each time.
  Mask &= ~(TRI->getSubRegIndexLaneMask(BestIdx));
  while (Mask.any()) {
    BestIdx = 0;
    int BestCover = std::numeric_limits<int>::min();
    for (unsigned Idx : PossibleIndexes) {
      LaneBitmask SubRegMask = TRI->getSubRegIndexLaneMask(Idx);
      // Early exit if we found a perfect match.
      if (SubRegMask == Mask) {
        BestIdx = Idx;
        break;
      }

      // Guaranteed above
      assert((SubRegMask & Avoid).none());

      // Try to cover as much of the remaining lanes as possible but as few of
      // the already covered lanes as possible.
      int Cover = (SubRegMask & Mask).getNumLanes() -
                  (SubRegMask & ~Mask).getNumLanes();
      if (Cover > BestCover) {
        BestCover = Cover;
        BestIdx = Idx;
      }
    }

    if (BestIdx == 0) {
      LLVM_DEBUG(
          dbgs() << "Unable to find minimal spanning sub register(s) for "
                 << TRI->getRegClassName(RC) << " mask " << PrintLaneMask(Mask)
                 << '\n');
      assert(false && "Impossible to span reg class");
      return std::vector<unsigned>();
    }

    Result.push_back(BestIdx);
    Mask &= ~TRI->getSubRegIndexLaneMask(BestIdx);
  }

  return Result;
}

static void updateSubReg(MachineOperand &UseMO,
                         const llvm::TargetRegisterClass *NewRC,
                         unsigned Offset, const SIRegisterInfo *SIRI) {
  unsigned Size = NewRC->getLaneMask().getNumLanes();
  if (Size == 1) {
    UseMO.setSubReg(0);
  } else {
    const uint32_t SubReg = UseMO.getSubReg();
    LaneBitmask LaneMask = SIRI->getSubRegIndexLaneMask(SubReg);

    unsigned Mask = LaneMask.getAsInteger() >> Offset;

    unsigned NewSubReg = getMinimalSpanningSubRegIdxSetForLaneMask(
                             SIRI, NewRC, LaneBitmask(Mask))
                             .front();

    UseMO.setSubReg(NewSubReg);
  }
}

bool reduceChannel(unsigned Offset, MachineInstr &MI, const MCInstrDesc &Desc,
                   MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                   const SIInstrInfo *SIII, SlotIndexes *SlotIndexes) {
  MachineOperand &DstMO = MI.getOperand(0);
  // Skip case when dst subReg not 0.
  if (DstMO.getSubReg()) {
    return false;
  }
  Register Reg = DstMO.getReg();

  SmallVector<MachineOperand *, 2> UseMOs;
  for (MachineOperand &UseMO : MRI.use_nodbg_operands(Reg)) {
    UseMOs.emplace_back(&UseMO);
  }

  const llvm::TargetRegisterClass *NewRC =
      SIRI->getRegClass(Desc.operands().front().RegClass);
  if (!NewRC->isAllocatable()) {
    if (SIRI->isSGPRClass(NewRC))
      NewRC = SIRI->getSGPRClassForBitWidth(NewRC->MC->RegSizeInBits);
    else if (SIRI->isVGPRClass(NewRC))
      NewRC = SIRI->getVGPRClassForBitWidth(NewRC->MC->RegSizeInBits);
    else
      return false;

    if (!NewRC->isAllocatable())
      return false;
  }

  unsigned NumLanes = NewRC->getLaneMask().getNumLanes();
  if (Offset > 0) {
    // Update offset operand in MI.
    MachineOperand *OffsetOp =
        SIII->getNamedOperand(MI, AMDGPU::OpName::offset);

    const uint32_t LaneSize = sizeof(uint32_t);
    if (OffsetOp) {
      if (OffsetOp->isImm()) {
        assert(OffsetOp != nullptr);
        int64_t Offset = OffsetOp->getImm();
        Offset += Offset * LaneSize;
        if (!SIII->isLegalMUBUFImmOffset(Offset)) {
          return false;
        }
        OffsetOp->setImm(Offset);
      } else {
        return false;
      }
    } else {
      OffsetOp = SIII->getNamedOperand(MI, AMDGPU::OpName::soffset);
      if (OffsetOp) {
        Register NewOffsetReg =
            MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
        auto OffsetAdd = BuildMI(*MI.getParent()->getParent(), MI.getDebugLoc(),
                                 SIII->get(AMDGPU::S_ADD_U32))
                             .addDef(NewOffsetReg)
                             .add(*OffsetOp)
                             .addImm(Offset * LaneSize);
        MachineInstr *OffsetAddMI = OffsetAdd.getInstr();
        MachineBasicBlock::iterator InsertPoint =
            llvm::findOrCreateInsertionPointForSccDef(MI.getParent(), MI, SIRI,
                                                      SIII, &MRI);
        MI.getParent()->insert(InsertPoint, OffsetAddMI);
        SIII->legalizeOperands(*OffsetAddMI);
        OffsetOp->setReg(NewOffsetReg);
        OffsetOp->setSubReg(0);
        if (SlotIndexes)
          SlotIndexes->insertMachineInstrInMaps(*OffsetAddMI);
      } else {
        return false;
      }
    }
    // Update subReg for users.
    for (MachineOperand *UseMO : UseMOs) {
      updateSubReg(*UseMO, NewRC, Offset, SIRI);
    }
  } else if (NumLanes == getNumLanesIn32BitReg(Reg, SIRI, MRI)) {
    // Clear subReg when it's a single 32-bit reg.
    for (MachineOperand *UseMO : UseMOs) {
      UseMO->setSubReg(0);
    }
  }

  MI.setDesc(Desc);
  // Mutate reg class of Reg.
  MRI.setRegClass(Reg, NewRC);
  return true;
}

bool removeUnusedLanes(llvm::MachineInstr &MI, MachineRegisterInfo &MRI,
                       const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
                       SlotIndexes *SlotIndexes) {
  bool IsImm = false;
  switch (MI.getOpcode()) {
  default:
    break;
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX16_IMM:
    IsImm = true;
    LLVM_FALLTHROUGH;
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_SGPR:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_SGPR:
  case AMDGPU::S_BUFFER_LOAD_DWORDX8_SGPR:
  case AMDGPU::S_BUFFER_LOAD_DWORDX16_SGPR: {
    Register Reg = MI.getOperand(0).getReg();
    if (!MRI.getUniqueVRegDef(Reg))
      return false;
    LaneBitmask DstMask = getRegMask(MI.getOperand(0), MRI);
    LaneBitmask UseMask;
    for (MachineOperand &MO : MRI.use_operands(Reg)) {
      UseMask |= llvm::getRegMask(MO, MRI);
    }

    const unsigned FullMask = DstMask.getAsInteger();
    unsigned Mask = UseMask.getAsInteger();
    if (Mask == FullMask)
      return false;
    // Split mask when there's gap. Then group mask to 2/4/8.
    auto Pieces = Piece::split(std::bitset<32>(Mask));
    // Now only support 1 piece.
    if (Pieces.size() != 1)
      return false;
    auto Piece = Pieces[0];
    if (Piece.Size > 8)
      return false;

    // TODO: enable offset support when IsImm is true.
    // Now if break different test when mul LaneSize or not mul for the offset.
    if (IsImm && Piece.Offset != 0)
      return false;

    const unsigned Num32BitLanes =
        Piece.Size / getNumLanesIn32BitReg(Reg, SIRI, MRI);

    switch (Num32BitLanes) {
    default:
      return false;
    case 1:
      return reduceChannel(Piece.Offset, MI,
                           SIII->get(IsImm ? AMDGPU::S_BUFFER_LOAD_DWORD_IMM
                                           : AMDGPU::S_BUFFER_LOAD_DWORD_SGPR),
                           MRI, SIRI, SIII, SlotIndexes);
    case 2:
      return reduceChannel(Piece.Offset, MI,
                           SIII->get(IsImm
                                         ? AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM
                                         : AMDGPU::S_BUFFER_LOAD_DWORDX2_SGPR),
                           MRI, SIRI, SIII, SlotIndexes);
    case 3:
      if (FullMask == 0xff)
        return false;
      LLVM_FALLTHROUGH;
    case 4:
      return reduceChannel(Piece.Offset, MI,
                           SIII->get(IsImm
                                         ? AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM
                                         : AMDGPU::S_BUFFER_LOAD_DWORDX4_SGPR),
                           MRI, SIRI, SIII, SlotIndexes);
    case 5:
    case 6:
    case 7:
      if (FullMask == 0xffff)
        return false;
      LLVM_FALLTHROUGH;
    case 8:
      return reduceChannel(Piece.Offset, MI,
                           SIII->get(IsImm
                                         ? AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM
                                         : AMDGPU::S_BUFFER_LOAD_DWORDX8_SGPR),
                           MRI, SIRI, SIII, SlotIndexes);
    }

  } break;
  }
  return false;
}

unsigned getRegSize(unsigned Reg, llvm::LaneBitmask &Mask,
                    const llvm::MachineRegisterInfo &MRI,
                    const llvm::SIRegisterInfo *SIRI) {
  unsigned Size = SIRI->getRegSizeInBits(*MRI.getRegClass(Reg));
  Size >>= 5;
  if (Mask.any()) {
    if (unsigned MaskSize = Mask.getNumLanes()) {
      if (MaskSize < Size)
        Size = MaskSize;
    }
  }
  return Size;
}

void collectLiveSetPressure(const LiveSet &LiveSet,
                            const MachineRegisterInfo &MRI,
                            const SIRegisterInfo *SIRI, unsigned &VPressure,
                            unsigned &SPressure) {
  VPressure = 0;
  SPressure = 0;
  for (auto LiveIt : LiveSet) {
    unsigned Reg = LiveIt.first;
    unsigned Size = getRegSize(Reg, LiveIt.second, MRI, SIRI);
    if (SIRI->isVGPR(MRI, Reg)) {
      VPressure += Size;
    } else {
      SPressure += Size;
    }
  }
}

bool isSub0Sub1SingleDef(unsigned Reg, const MachineRegisterInfo &MRI) {
  // Support multi def for pattern of pointer:
  // undef_ %808.sub0:sgpr_64 = COPY killed %795:sgpr_32
  // %808.sub1:sgpr_64 = S_MOV_B32 0
  bool HasSub0 = false;
  bool HasSub1 = false;
  for (MachineOperand &UserDefMO : MRI.def_operands(Reg)) {
    if (unsigned SubReg = UserDefMO.getSubReg()) {
      bool IsSingleSubReg = false;
      switch (SubReg) {
      default:
        break;
      case AMDGPU::sub0:
        if (!HasSub0) {
          HasSub0 = true;
          IsSingleSubReg = true;
        }
        break;
      case AMDGPU::sub1:
        if (!HasSub1) {
          HasSub1 = true;
          IsSingleSubReg = true;
        }
        break;
      }
      if (!IsSingleSubReg) {
        HasSub0 = false;
        break;
      }
    } else {
      HasSub0 = false;
      break;
    }
  }

  return (HasSub0 && HasSub1);
}

bool reach_block(MachineBasicBlock *FromBB, MachineDominatorTree *DT,
                 MachinePostDominatorTree *PDT, MachineLoopInfo *LI,
                 MachineBasicBlock *ToBB) {
  if (FromBB == ToBB) {
    return true;
  }

  if (DT->dominates(FromBB, ToBB)) {
    return true;
  }

  if (PDT->dominates(ToBB, FromBB)) {
    return true;
  }

  if (loopContainsBoth(LI, ToBB, FromBB)) {
    return true;
  }
  // TODO: cover case hotBB in loop,
  //       one block in that loop dom BB or
  //       BB post dom one block in that loop.
  return false;
}
} // namespace llvm
