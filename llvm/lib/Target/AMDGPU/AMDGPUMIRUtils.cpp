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

#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/SlotIndexes.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/Support/GraphWriter.h"

#include "llvm/Support/Debug.h"

#include "AMDGPUMIRUtils.h"
#include "AMDGPUSubExpDag.h"
#include "GCNRegPressure.h"
#include <unordered_set>

#define DEBUG_TYPE "xb-mir-util"
using namespace llvm;
namespace {
class CFGWithPhi {
public:
  CFGWithPhi(MachineFunction &F) : F(F) {
    // Collect phi and phi related insts.
    MachineRegisterInfo &MRI = F.getRegInfo();

    for (MachineBasicBlock &BB : F) {
      auto &PhiInsts = BlockToPhiInstsMap[&BB];
      for (MachineInstr &I : BB) {
        if (!I.isPHI())
          break;
        PhiInsts.insert(&I);
        Register Reg = I.getOperand(0).getReg();
        // Add incoming values.
        for (unsigned i = 1; i < I.getNumOperands(); i += 2) {
          MachineOperand &MO = I.getOperand(i);
          if (!MO.isReg())
            continue;
          MachineInstr *DefMI = MRI.getUniqueVRegDef(MO.getReg());
          if (!DefMI)
            continue;
          BlockToPhiInstsMap[DefMI->getParent()].insert(DefMI);
        }
        // Add users.
        for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
          BlockToPhiInstsMap[UseMI.getParent()].insert(&UseMI);
        }
      }
    }
  } /// Adds custom features for a visualization of the ScheduleDAG.
  void addCustomGraphFeatures(llvm::GraphWriter<CFGWithPhi *> &) const {}
  MachineFunction &F;
  DenseMap<const MachineBasicBlock *, DenseSet<MachineInstr *>>
      BlockToPhiInstsMap;
  void dump();
};

void CFGWithPhi::dump() {
#ifndef NDEBUG
  for (MachineBasicBlock &BB : F) {
    dbgs() << BB.getName() << "\n";
    auto &PhiInsts = BlockToPhiInstsMap[&BB];
    for (MachineInstr *I : PhiInsts) {
      if (!I->isPHI())
        continue;
      I->dump();
    }
    for (MachineInstr *I : PhiInsts) {
      if (I->isPHI())
        continue;
      I->dump();
    }
  }
#endif
}

} // namespace

// CFGWithPhi dump.
namespace llvm {

template <> struct DOTGraphTraits<CFGWithPhi *> : public DefaultDOTGraphTraits {

  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(const CFGWithPhi *) {
    return "CFG with Phi graph";
  }

  static std::string getNodeIdentifierLabel(const MachineBasicBlock *Node,
                                            const CFGWithPhi *) {
    std::string R;
    raw_string_ostream OS(R);
    OS << static_cast<const void *>(Node);
    return R;
  }

  static std::string getNodeLabel(const MachineBasicBlock *BB,
                                  const CFGWithPhi *G) {
    enum { MaxColumns = 8000 };
    std::string Str;
    raw_string_ostream OS(Str);

    OS << "BB:" << BB->getName();
    auto It = G->BlockToPhiInstsMap.find(BB);
    if (It != G->BlockToPhiInstsMap.end()) {

      auto &PhiInsts = It->second;
      for (MachineInstr *I : PhiInsts) {
        if (!I->isPHI())
          continue;
        I->print(OS);
        OS << "\n";
      }
      for (MachineInstr *I : PhiInsts) {
        if (I->isPHI())
          continue;
        I->print(OS);
        OS << "\n";
      }
    }
    std::string OutStr = OS.str();
    if (OutStr[0] == '\n')
      OutStr.erase(OutStr.begin());

    // Process string output to make it nicer...
    unsigned ColNum = 0;
    unsigned LastSpace = 0;
    for (unsigned i = 0; i != OutStr.length(); ++i) {
      if (OutStr[i] == '\n') { // Left justify
        OutStr[i] = '\\';
        OutStr.insert(OutStr.begin() + i + 1, 'l');
        ColNum = 0;
        LastSpace = 0;
      } else if (OutStr[i] == ';') {             // Delete comments!
        unsigned Idx = OutStr.find('\n', i + 1); // Find end of line
        OutStr.erase(OutStr.begin() + i, OutStr.begin() + Idx);
        --i;
      } else if (ColNum == MaxColumns) { // Wrap lines.
        // Wrap very long names even though we can't find a space.
        if (!LastSpace)
          LastSpace = i;
        OutStr.insert(LastSpace, "\\l...");
        ColNum = i - LastSpace;
        LastSpace = 0;
        i += 3; // The loop will advance 'i' again.
      } else
        ++ColNum;
      if (OutStr[i] == ' ')
        LastSpace = i;
    }
    return OutStr;
  }
  static std::string getNodeDescription(const MachineBasicBlock *SU,
                                        const CFGWithPhi *) {
    return SU->getName().str();
  }

  static void addCustomGraphFeatures(CFGWithPhi *G,
                                     GraphWriter<CFGWithPhi *> &GW) {
    return G->addCustomGraphFeatures(GW);
  }
};

template <> struct GraphTraits<CFGWithPhi *> {
  using NodeRef = MachineBasicBlock *;
  using ChildIteratorType = MachineBasicBlock::succ_iterator;
  using nodes_iterator = pointer_iterator<MachineFunction::iterator>;

  // static NodeRef getEntryNode(const CFGWithPhi *G) {
  //  return G->F.getFunctionEntry();
  //}

  static ChildIteratorType child_begin(const NodeRef N) {
    return N->succ_begin();
  }

  static ChildIteratorType child_end(const NodeRef N) { return N->succ_end(); }

  static nodes_iterator nodes_begin(const CFGWithPhi *G) {
    return nodes_iterator(G->F.begin());
  }

  static nodes_iterator nodes_end(const CFGWithPhi *G) {
    return nodes_iterator(G->F.end());
  }
};

} // namespace llvm

namespace llvm {

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

bool isExecUpdateForControlFlow(llvm::MachineInstr &MI) {
  bool IsExecUpdate = false;
  unsigned Opcode = MI.getOpcode();
  if (Opcode == AMDGPU::S_MOV_B64 || Opcode == AMDGPU::S_MOV_B32 ||
      Opcode == AMDGPU::S_OR_B64_term || Opcode == AMDGPU::S_OR_B32_term ||
      Opcode == AMDGPU::S_OR_SAVEEXEC_B64 ||
      Opcode == AMDGPU::S_OR_SAVEEXEC_B32 || Opcode == AMDGPU::S_AND_B64 ||
      Opcode == AMDGPU::S_AND_B32 || Opcode == AMDGPU::S_ANDN2_B64 ||
      Opcode == AMDGPU::S_ANDN2_B32) {
    MachineOperand &Dst = MI.getOperand(0);
    if (Dst.getReg() == AMDGPU::EXEC || Dst.getReg() == AMDGPU::EXEC_LO) {
      IsExecUpdate = true;
    }
  }
  return IsExecUpdate;
}

bool isSub0Sub1SingleDef(unsigned Reg, const MachineRegisterInfo &MRI) {
  // Support multi def for pattern of pointer:
  // undef %808.sub0:sgpr_64 = COPY killed %795:sgpr_32
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

LaneBitmask getRegMask(const MachineOperand &MO,
                       const MachineRegisterInfo &MRI) {
  // We don't rely on read-undef flag because in case of tentative schedule
  // tracking it isn't set correctly yet. This works correctly however since
  // use mask has been tracked before using LIS.
  return MO.getSubReg() == 0
             ? MRI.getMaxLaneMaskForVReg(MO.getReg())
             : MRI.getTargetRegisterInfo()->getSubRegIndexLaneMask(
                   MO.getSubReg());
}

void mergeLiveRegSet(LiveSet &TargetSet, const LiveSet &InputSet) {
  for (auto It : InputSet) {
    Register Reg = It.first;
    LaneBitmask Mask = It.second;
    auto TargetReg = TargetSet.find(Reg);
    if (TargetReg != TargetSet.end()) {
      LaneBitmask TargetMask = TargetReg->second;
      Mask |= TargetMask;
    }
    TargetSet[Reg] = Mask;
  }
}

void andLiveRegSet(LiveSet &TargetSet, const LiveSet &InputSet) {
  GCNRPTracker::LiveRegSet AndSet;
  for (auto It : InputSet) {
    Register Reg = It.first;
    LaneBitmask Mask = It.second;
    auto TargetReg = TargetSet.find(Reg);
    if (TargetReg != TargetSet.end()) {
      LaneBitmask TargetMask = TargetReg->second;
      Mask &= TargetMask;
      AndSet[Reg] = Mask;
    }
  }

  TargetSet = AndSet;
}

void andNotLiveRegSet(LiveSet &TargetSet, const LiveSet &InputSet) {
  for (auto It : InputSet) {
    unsigned Reg = It.first;
    LaneBitmask Mask = It.second;
    auto TargetReg = TargetSet.find(Reg);
    if (TargetReg != TargetSet.end()) {
      LaneBitmask TargetMask = TargetReg->second;
      if ((TargetMask | Mask) == Mask)
        TargetSet.erase(Reg);
      else
        TargetSet[Reg] = TargetMask & (~Mask);
    }
  }
}

MachineBasicBlock *split(MachineInstr *Inst) {

  // Create the fall-through block.
  MachineBasicBlock *MBB = Inst->getParent();
  MachineFunction *MF = MBB->getParent();
  MachineBasicBlock *SuccMBB = MF->CreateMachineBasicBlock();
  auto MBBIter = ++(MBB->getIterator());
  MF->insert(MBBIter, SuccMBB);
  SuccMBB->transferSuccessorsAndUpdatePHIs(MBB);
  MBB->addSuccessor(SuccMBB);

  // Splice the code over.
  SuccMBB->splice(SuccMBB->end(), MBB, ++Inst->getIterator(), MBB->end());

  return SuccMBB;
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
  unsigned Size = NewRC->getLaneMask().getNumLanes();
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
  } else if (Size == 1) {
    // Clear subReg when size is 1.
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

    switch (Piece.Size) {
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
      if (FullMask == 0xf)
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
      if (FullMask == 0xff)
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

// True if there is a loop which contains both BB1 and BB2.
bool loopContainsBoth(const MachineLoopInfo *LI, const MachineBasicBlock *BB1,
                      const MachineBasicBlock *BB2) {
  const MachineLoop *L1 = getOutermostLoop(LI, BB1);
  const MachineLoop *L2 = getOutermostLoop(LI, BB2);
  return L1 != nullptr && L1 == L2;
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

// If BB can reach hotMBBs.
bool reach_blocks(MachineBasicBlock *BB, MachineDominatorTree *DT,
                  MachinePostDominatorTree *PDT, MachineLoopInfo *LI,
                  DenseSet<MachineBasicBlock *> &HotMBBs) {
  bool Cross = false;
  for (MachineBasicBlock *HotBB : HotMBBs) {
    if (reach_block(BB, DT, PDT, LI, HotBB)) {
      Cross = true;
      break;
    }
  }
  return Cross;
}

} // namespace llvm

namespace llvm {
void viewCFGWithPhi(llvm::MachineFunction &F) {
#ifdef DBG
  CFGWithPhi G(F);
  ViewGraph(const_cast<CFGWithPhi *>(&G), F.getName(), false, F.getName());
  G.dump();
#endif
}
} // namespace llvm

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

// Helper functions to Write jason.
namespace {
void json_name(StringRef Val, raw_ostream &OS) { OS << "\"" << Val << "\":"; }

template <typename write_fn>
void json_pair(StringRef Val, write_fn &Fn, raw_ostream &OS) {
  json_name(Val, OS);
  OS << "\"";
  Fn();
  OS << "\"";
}

template <typename write_fn>
void json_obj_pair(StringRef Val, write_fn &Fn, raw_ostream &OS) {
  json_name(Val, OS);

  Fn();
}

template <typename write_fn>
void json_array(StringRef Val, write_fn &Fn, raw_ostream &OS) {
  json_name(Val, OS);
  OS << "[";
  Fn();
  OS << "]";
}
} // namespace

namespace llvm {
namespace pressure {

void write_inst(MachineInstr &MI, const SlotIndexes *SlotIndexes,
                const SIInstrInfo *SIII, raw_ostream &OS) {
  OS << "{";
  SlotIndex Slot = SlotIndexes->getInstructionIndex(MI);
  auto WriteSlot = [&Slot, &OS]() { Slot.print(OS); };

  json_pair("slot_index", WriteSlot, OS);

  OS << ",";

  auto WriteOpcode = [&MI, &SIII, &OS]() {
    OS << SIII->getName(MI.getOpcode());
  };

  json_pair("opcode", WriteOpcode, OS);

  OS << ",";

  auto WriteAsm = [&MI, &SIII, &OS]() {
    MI.print(OS, /*IsStandalone*/ true, /*SkipOpers*/ false,
             /*SkipDebugLoc*/ true, /*AddNewLine*/ false, SIII);
  };
  json_pair("asm", WriteAsm, OS);

  OS << "}";
}

void print_reg(Register Reg, const MachineRegisterInfo &MRI,
               const SIRegisterInfo *SIRI, raw_ostream &OS) {
  if (Reg.isVirtual()) {
    StringRef Name = MRI.getVRegName(Reg);
    if (Name != "") {
      OS << '%' << Name;
    } else {
      OS << '%' << Register::virtReg2Index(Reg);
    }
  } else if (Reg < SIRI->getNumRegs()) {
    OS << '$';
    printLowerCase(SIRI->getName(Reg), OS);
  } else {
    llvm_unreachable("invalid reg");
  }
}

void write_reg(unsigned Reg, unsigned SubReg, const MachineRegisterInfo &MRI,
               const SIRegisterInfo *SIRI, raw_ostream &OS) {
  OS << "{";

  auto WriteReg = [&MRI, &SIRI, &Reg, &OS]() { print_reg(Reg, MRI, SIRI, OS); };
  json_pair("reg", WriteReg, OS);

  OS << ",";

  auto WriteSubReg = [&SubReg, &OS]() { OS << SubReg; };

  json_pair("sub_reg", WriteSubReg, OS);

  OS << ",";
  auto WriteIsSgpr = [&Reg, &MRI, &SIRI, &OS]() {
    if (SIRI->isSGPRReg(MRI, Reg))
      OS << "true";
    else
      OS << "false";
  };
  json_obj_pair("is_sgpr", WriteIsSgpr, OS);
  OS << "}";
}

unsigned get_reg_size(unsigned Reg, const MachineRegisterInfo &MRI,
                      const SIRegisterInfo *SIRI) {
  return SIRI->getRegClassForReg(MRI, Reg)->getLaneMask().getNumLanes();
}

void write_live(unsigned Reg, LaneBitmask Mask, const MachineRegisterInfo &MRI,
                const SIRegisterInfo *SIRI, raw_ostream &OS) {
  if (Mask.none()) {
    unsigned Size = get_reg_size(Reg, MRI, SIRI);
    Mask = LaneBitmask((1 << Size) - 1);
  }
  unsigned IntMask = Mask.getAsInteger();
  for (unsigned i = 0; i <= Mask.getHighestLane(); i++) {
    if (IntMask & (1 << i)) {
      write_reg(Reg, i, MRI, SIRI, OS);
      OS << ",\n";
    }
  }
}

void write_dag_input_node(unsigned ID, unsigned Reg, unsigned Mask,
                          const MachineRegisterInfo &MRI,
                          const SIRegisterInfo *SIRI, raw_ostream &OS) {
  OS << "{";
  auto WriteID = [&ID, &OS]() { OS << ID; };

  json_pair("ID", WriteID, OS);

  OS << ",";

  auto WriteReg = [&Reg, &MRI, &SIRI, &OS]() { print_reg(Reg, MRI, SIRI, OS); };

  json_pair("reg", WriteReg, OS);

  OS << ",";

  auto WriteMask = [&Mask, &OS]() { OS << Mask; };

  json_pair("mask", WriteMask, OS);

  OS << "},\n";
}

void write_dag_inst_node(unsigned ID, SlotIndex Slot,
                         GCNRPTracker::LiveRegSet LiveReg,
                         const MachineRegisterInfo &MRI,
                         const SIRegisterInfo *SIRI, SUnit *SU,
                         raw_ostream &OS) {
  OS << "{";
  auto WriteID = [&ID, &OS]() { OS << ID; };

  json_pair("ID", WriteID, OS);

  OS << ",";

  auto WriteSlot = [&Slot, &OS]() { Slot.print(OS); };

  json_pair("slot_index", WriteSlot, OS);

  OS << ",";

  auto WriteRegs = [&LiveReg, &MRI, &SIRI, &OS]() {
    for (auto It : LiveReg) {
      unsigned Reg = It.first;
      LaneBitmask Mask = It.second;
      write_live(Reg, Mask, MRI, SIRI, OS);
    }
  };
  json_array("regs", WriteRegs, OS);

  OS << ",";

  auto WritePreds = [&SU, &OS]() {
    for (auto &Pred : SU->Preds) {

      OS << Pred.getSUnit()->NodeNum << ",";
    }
  };

  json_array("preds", WritePreds, OS);

  OS << "},\n";
}

void write_block(MachineBasicBlock &Blk, LiveIntervals *LIS,
                 const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                 const SIInstrInfo *SIII, raw_ostream &OS) {
  OS << "{\n";
  auto WriteName = [&Blk, &OS]() { OS << Blk.getName(); };
  json_pair("name", WriteName, OS);

  OS << ",";

  auto WriteIndex = [&Blk, &OS]() { OS << Blk.getNumber(); };
  json_pair("id", WriteIndex, OS);

  OS << ",";

  const SlotIndexes *SlotIndexes = LIS->getSlotIndexes();

  SlotIndex BeginSlot = SlotIndexes->getMBBStartIdx(&Blk);
  auto WriteSlot = [&BeginSlot, &OS]() { BeginSlot.print(OS); };
  json_pair("begin_slot", WriteSlot, OS);

  OS << ",";

  SlotIndex EndSlot = SlotIndexes->getMBBEndIdx(&Blk);
  auto WriteEndSlot = [&EndSlot, &OS]() { EndSlot.print(OS); };
  json_pair("end_slot", WriteEndSlot, OS);

  OS << ",";

  auto WriteInsts = [&Blk, &SlotIndexes, &SIII, &OS]() {
    for (MachineInstr &MI : Blk) {
      if (MI.isDebugInstr())
        continue;
      write_inst(MI, SlotIndexes, SIII, OS);
      OS << ",\n";
    }
  };

  json_array("instructions", WriteInsts, OS);

  OS << ",";

  BlockExpDag Dag(&Blk, LIS, MRI, SIRI, SIII);
  Dag.buildWithPressure();

  const auto StartLiveReg = llvm::getLiveRegs(BeginSlot, *Dag.LIS, Dag.MRI);
  auto WriteInputs = [&StartLiveReg, &Dag, &OS]() {
    for (auto It : StartLiveReg) {
      unsigned Reg = It.first;
      LaneBitmask Mask = It.second;
      SUnit *SU = Dag.InputSUnitMap[Reg];
      // Write Reg and mask to the nodes.
      write_dag_input_node(SU->NodeNum, Reg, Mask.getAsInteger(), Dag.MRI,
                           Dag.SIRI, OS);
    }
  };

  json_array("input_nodes", WriteInputs, OS);

  OS << ",";

  auto WriteNodes = [&SlotIndexes, &Dag, &OS]() {
    for (auto It : Dag.MISUnitMap) {
      MachineInstr *MI = It.first;
      SUnit *SU = It.second;
      // Use SlotIndex of MI.
      SlotIndex SlotIndex;
      if (!MI->isDebugInstr())
        SlotIndex = SlotIndexes->getInstructionIndex(*MI);
      GCNRPTracker::LiveRegSet LiveReg = Dag.DagPressureMap[SU];
      // Write slot, live to the nodes.
      write_dag_inst_node(SU->NodeNum, SlotIndex, LiveReg, Dag.MRI, Dag.SIRI,
                          SU, OS);
    }
  };

  json_array("inst_nodes", WriteNodes, OS);

  OS << ",";

  auto WritePreds = [&Blk, &OS]() {
    for (MachineBasicBlock *Pred : Blk.predecessors()) {
      OS << Pred->getNumber() << ",";
    }
  };

  json_array("preds", WritePreds, OS);

  OS << ",";

  auto WriteSuccs = [&Blk, &OS]() {
    for (MachineBasicBlock *Succ : Blk.successors()) {
      OS << Succ->getNumber() << ",";
    }
  };

  json_array("succs", WriteSuccs, OS);

  OS << "}";
}

void write_define(SlotIndex &Slot, unsigned Reg, unsigned SubReg,
                  const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                  raw_ostream &OS) {
  OS << "{";
  auto WriteSlot = [&Slot, &OS]() { Slot.print(OS); };

  json_pair("slot_index", WriteSlot, OS);

  OS << ",";

  auto WriteReg = [&MRI, &SIRI, &Reg, &SubReg, &OS]() {
    write_reg(Reg, SubReg, MRI, SIRI, OS);
  };
  json_obj_pair("reg", WriteReg, OS);

  OS << "}\n";

  OS << ",";
}

void write_define(MachineOperand &MO, const SlotIndexes *SlotIndexes,
                  const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                  raw_ostream &OS) {
  // Split subReg?  MO.getSubReg();
  Register Reg = MO.getReg();
  unsigned SubReg = MO.getSubReg();
  MachineInstr *MI = MO.getParent();
  SlotIndex Slot = SlotIndexes->getInstructionIndex(*MI);
  if (SubReg == 0) {
    unsigned Size = get_reg_size(Reg, MRI, SIRI);
    for (unsigned i = 0; i < Size; i++) {
      write_define(Slot, Reg, i, MRI, SIRI, OS);
    }
  } else {
    switch (SubReg) {
    default:
      assert(0 && "SubReg not supported yet.");
      write_define(Slot, Reg, SubReg, MRI, SIRI, OS);
      break;
    case AMDGPU::sub0:
      write_define(Slot, Reg, 0, MRI, SIRI, OS);
      break;
    case AMDGPU::sub1:
      write_define(Slot, Reg, 1, MRI, SIRI, OS);
      break;
    case AMDGPU::sub2:
      write_define(Slot, Reg, 2, MRI, SIRI, OS);
      break;
    case AMDGPU::sub3:
      write_define(Slot, Reg, 3, MRI, SIRI, OS);
      break;
    case AMDGPU::sub4:
      write_define(Slot, Reg, 4, MRI, SIRI, OS);
      break;
    case AMDGPU::sub5:
      write_define(Slot, Reg, 5, MRI, SIRI, OS);
      break;
    case AMDGPU::sub6:
      write_define(Slot, Reg, 6, MRI, SIRI, OS);
      break;
    case AMDGPU::sub7:
      write_define(Slot, Reg, 7, MRI, SIRI, OS);
      break;
    case AMDGPU::sub8:
      write_define(Slot, Reg, 8, MRI, SIRI, OS);
      break;
    case AMDGPU::sub9:
      write_define(Slot, Reg, 9, MRI, SIRI, OS);
      break;
    case AMDGPU::sub10:
      write_define(Slot, Reg, 10, MRI, SIRI, OS);
      break;
    case AMDGPU::sub11:
      write_define(Slot, Reg, 11, MRI, SIRI, OS);
      break;
    case AMDGPU::sub12:
      write_define(Slot, Reg, 12, MRI, SIRI, OS);
      break;
    case AMDGPU::sub13:
      write_define(Slot, Reg, 13, MRI, SIRI, OS);
      break;
    case AMDGPU::sub14:
      write_define(Slot, Reg, 14, MRI, SIRI, OS);
      break;
    case AMDGPU::sub15:
      write_define(Slot, Reg, 15, MRI, SIRI, OS);
      break;
    case AMDGPU::sub0_sub1:
      write_define(Slot, Reg, 0, MRI, SIRI, OS);
      write_define(Slot, Reg, 1, MRI, SIRI, OS);
      break;
    case AMDGPU::sub2_sub3:
      write_define(Slot, Reg, 2, MRI, SIRI, OS);
      write_define(Slot, Reg, 3, MRI, SIRI, OS);
      break;
    case AMDGPU::sub4_sub5:
      write_define(Slot, Reg, 4, MRI, SIRI, OS);
      write_define(Slot, Reg, 5, MRI, SIRI, OS);
      break;
    case AMDGPU::sub1_sub2:
      write_define(Slot, Reg, 1, MRI, SIRI, OS);
      write_define(Slot, Reg, 2, MRI, SIRI, OS);
      break;
    case AMDGPU::sub0_sub1_sub2:
      write_define(Slot, Reg, 0, MRI, SIRI, OS);
      write_define(Slot, Reg, 1, MRI, SIRI, OS);
      write_define(Slot, Reg, 2, MRI, SIRI, OS);
      break;
    case AMDGPU::sub0_sub1_sub2_sub3:
      write_define(Slot, Reg, 0, MRI, SIRI, OS);
      write_define(Slot, Reg, 1, MRI, SIRI, OS);
      write_define(Slot, Reg, 2, MRI, SIRI, OS);
      write_define(Slot, Reg, 3, MRI, SIRI, OS);
      break;
    case AMDGPU::sub2_sub3_sub4_sub5:
      write_define(Slot, Reg, 2, MRI, SIRI, OS);
      write_define(Slot, Reg, 3, MRI, SIRI, OS);
      write_define(Slot, Reg, 4, MRI, SIRI, OS);
      write_define(Slot, Reg, 5, MRI, SIRI, OS);
      break;
    case AMDGPU::sub0_sub1_sub2_sub3_sub4_sub5_sub6_sub7:
      write_define(Slot, Reg, 0, MRI, SIRI, OS);
      write_define(Slot, Reg, 1, MRI, SIRI, OS);
      write_define(Slot, Reg, 2, MRI, SIRI, OS);
      write_define(Slot, Reg, 3, MRI, SIRI, OS);
      write_define(Slot, Reg, 4, MRI, SIRI, OS);
      write_define(Slot, Reg, 5, MRI, SIRI, OS);
      write_define(Slot, Reg, 6, MRI, SIRI, OS);
      write_define(Slot, Reg, 7, MRI, SIRI, OS);
      break;
    }
  }
}

void write_defines(MachineFunction &MF, const SlotIndexes *SlotIndexes,
                   const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                   raw_ostream &OS) {

  for (unsigned i = 0; i < MRI.getNumVirtRegs(); i++) {
    auto Reg = Register::index2VirtReg(i);

    for (MachineOperand &MO : MRI.def_operands(Reg)) {
      write_define(MO, SlotIndexes, MRI, SIRI, OS);
    }
  }
}

void write_uses(MachineFunction &MF, const SlotIndexes *SlotIndexes,

                const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                raw_ostream &OS) {

  for (unsigned i = 0; i < MRI.getNumVirtRegs(); i++) {
    auto Reg = Register::index2VirtReg(i);

    for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
      // TODO: create write_use if use has more info.
      write_define(MO, SlotIndexes, MRI, SIRI, OS);
    }
  }
}

void write_liveness(SlotIndex Slot, GCNRPTracker::LiveRegSet &LiveSet,
                    const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                    raw_ostream &OS) {
  OS << "{";
  auto WriteSlot = [&Slot, &OS]() { Slot.print(OS); };

  json_pair("slot_index", WriteSlot, OS);

  OS << ",";

  auto WriteRegs = [&LiveSet, &MRI, &SIRI, &OS]() {
    for (auto it : LiveSet) {
      unsigned Reg = it.first;
      LaneBitmask Mask = it.second;
      write_live(Reg, Mask, MRI, SIRI, OS);
    }
  };
  json_array("regs", WriteRegs, OS);
  OS << "\n},\n";
}

void write_segment(const LiveInterval::Segment &S, raw_ostream &OS) {
  OS << "{";
  auto WriteBegin = [&S, &OS]() { S.start.print(OS); };

  json_pair("begin", WriteBegin, OS);

  OS << ",";

  auto WriteEnd = [&S, &OS]() { S.end.print(OS); };

  json_pair("end", WriteEnd, OS);

  OS << ",";

  auto WriteValNum = [&S, &OS]() {
    if (S.valno)
      OS << S.valno->id;
    else
      OS << 0xFFFFFFFF;
  };

  json_pair("val_num", WriteValNum, OS);

  OS << "},\n";
}

void write_subrange(const LiveInterval::SubRange &SR, raw_ostream &OS) {
  OS << "{\n";
  auto WriteMask = [&SR, &OS]() { OS << SR.LaneMask.getAsInteger(); };

  json_pair("mask", WriteMask, OS);

  OS << ",";

  // Segments.
  auto WriteSegments = [&SR, &OS]() {
    for (auto &S : SR.segments) {
      write_segment(S, OS);
    }
  };

  json_array("segments", WriteSegments, OS);

  OS << "\n},\n";
}

void write_live_interval(LiveInterval &LI, const MachineRegisterInfo &MRI,
                         const SIRegisterInfo *SIRI, raw_ostream &OS) {
  OS << "{\n";

  auto WriteReg = [&LI, &MRI, &SIRI, &OS]() {
    write_reg(LI.reg(), 0, MRI, SIRI, OS);
  };

  json_obj_pair("reg", WriteReg, OS);

  OS << ",";

  auto WriteSegments = [&LI, &OS]() {
    for (auto &S : LI.segments) {
      write_segment(S, OS);
    }
  };

  json_array("segments", WriteSegments, OS);

  OS << ",";

  auto WriteSubRanges = [&LI, &OS]() {
    for (auto &SR : LI.subranges()) {
      write_subrange(SR, OS);
    }
  };

  json_array("subranges", WriteSubRanges, OS);

  OS << "},\n";
}

std::string get_legal_str(const MDString *MDStr) {
  std::string Str;
  raw_string_ostream Stream(Str);
  MDStr->print(Stream);
  Stream.flush();
  // Remove !.
  Str = Str.substr(1);
  // Remove ""
  Str = Str.substr(1);
  Str.pop_back();
  std::replace(Str.begin(), Str.end(), '\\', '#');
  return Str;
}

void write_file(const MDNode *FileNode, raw_ostream &OS) {
  const MDString *FileName = cast<MDString>(FileNode->getOperand(0).get());
  StringRef FileNameStr = FileName->getString();
  if (FileNameStr.find("__AMDGPU_GPUMAP_") == 0)
    return;
  if (FileNameStr.find("__AMDGPU_DWARF_") == 0)
    return;

  OS << "{";

  std::string Str0 = get_legal_str(FileName);
  auto WriteName = [&Str0, &OS]() { OS << Str0; };
  json_pair("filename", WriteName, OS);

  OS << ",\n";

  const MDString *Content = cast<MDString>(FileNode->getOperand(1).get());
  std::string Str = get_legal_str(Content);
  auto WriteContent = [&Str, &OS]() { OS << Str; };
  json_pair("content", WriteContent, OS);
  OS << "\n},\n";
}

void write_DIFile(const DIFile *File, raw_ostream &OS) {
  if (File) {
    std::string Name = get_legal_str(File->getRawFilename());
    std::string Dir = "";
    if (MDString *MDDir = File->getRawDirectory())
      Dir = get_legal_str(MDDir);
    OS << Dir << Name;
  } else {
    OS << "ArtificialFile";
  }
}

void write_line_mapping(SlotIndex Slot, DebugLoc DL, raw_ostream &OS) {
  OS << "{";

  auto WriteSlot = [&Slot, &OS]() { Slot.print(OS); };

  json_pair("slot_index", WriteSlot, OS);

  OS << ",\n";

  MDNode *Scope = DL.getScope();
  unsigned Line = DL.getLine();
  unsigned Col = DL.getCol();

  auto WriteLine = [&Line, &OS]() { OS << Line; };
  json_pair("line", WriteLine, OS);

  OS << ",\n";

  auto WriteCol = [&Col, &OS]() { OS << Col; };
  json_pair("col", WriteCol, OS);

  OS << ",\n";

  auto WriteFile = [&Scope, &OS]() {
    const DIFile *File = cast<DIScope>(Scope)->getFile();
    write_DIFile(File, OS);
  };
  json_pair("file", WriteFile, OS);

  if (DILocation *InlineDL = DL.getInlinedAt()) {
    OS << ",\n";
    unsigned InlineLine = InlineDL->getLine();
    auto WriteLine = [&InlineLine, &OS]() { OS << InlineLine; };
    json_pair("inline_line", WriteLine, OS);

    OS << ",\n";

    unsigned InlineCol = InlineDL->getColumn();
    auto WriteCol = [&InlineCol, &OS]() { OS << InlineCol; };
    json_pair("inline_col", WriteCol, OS);

    OS << ",\n";

    const MDNode *InlineScope = DL.getInlinedAtScope();
    auto WriteFile = [&InlineScope, &OS]() {
      const DIFile *File = cast<DIScope>(InlineScope)->getFile();
      write_DIFile(File, OS);
    };
    json_pair("inline_file", WriteFile, OS);
  }

  OS << "\n},\n";
}

void write_dbg_val(unsigned Reg, const DIVariable *V, const DIExpression *Exp,
                   const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                   raw_ostream &OS) {
  OS << "{";

  auto WriteReg = [&MRI, &SIRI, &Reg, &OS]() {
    const unsigned SubReg = 0;
    write_reg(Reg, SubReg, MRI, SIRI, OS);
  };
  json_obj_pair("reg", WriteReg, OS);

  OS << ",\n";

  if (V) {
    auto WriteName = [&V, &OS]() { OS << V->getName(); };
    json_pair("debug_val_name", WriteName, OS);
    OS << ",\n";

    auto WriteFile = [&V, &OS]() {
      const DIFile *File = V->getFile();
      write_DIFile(File, OS);
    };
    json_pair("debug_val_file", WriteFile, OS);
    OS << ",\n";

    auto WriteLine = [&V, &OS]() { OS << V->getLine(); };
    json_pair("debug_val_line", WriteLine, OS);
  }

  if (Exp->isValid() && Exp->getNumElements()) {
    OS << ",\n";
    auto WriteV = [&Exp, &OS]() {
      OS << '[';
      bool NeedSep = false;
      for (auto Op : Exp->expr_ops()) {
        if (NeedSep)
          OS << ", ";
        else
          NeedSep = true;
        OS << dwarf::OperationEncodingString(Op.getOp());
        for (unsigned I = 0; I < Op.getNumArgs(); ++I)
          OS << ' ' << Op.getArg(I);
      }
      OS << "] ";
    };
    json_pair("debug_exp", WriteV, OS);
  }
  OS << "\n},\n";
}

void write_dbg_info(MachineFunction &MF, LiveIntervals *LIS,
                    const MachineRegisterInfo &MRI, const SIInstrInfo *SIII,
                    const SIRegisterInfo *SIRI, const SlotIndexes *SlotIndexes,
                    const NamedMDNode *SourceMD, raw_ostream &OS) {
  OS << ",\n";

  auto WriteFiles = [&SourceMD, &OS]() {
    for (const MDNode *FileNode : SourceMD->operands()) {
      write_file(FileNode, OS);
    }
  };

  json_array("files", WriteFiles, OS);

  OS << ",\n";

  auto WriteLineMapping = [&MF, &SlotIndexes, &OS]() {
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (MI.isDebugInstr()) {
          continue;
        }
        const DebugLoc DL = MI.getDebugLoc();
        if (!DL)
          continue;
        SlotIndex Slot = SlotIndexes->getInstructionIndex(MI);
        write_line_mapping(Slot, DL, OS);
      }
    }
  };

  json_array("line_mapping", WriteLineMapping, OS);

  OS << ",\n";

  auto WriteDebugVals = [&MF, &MRI, &SIRI, &OS]() {
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!MI.isDebugValue())
          continue;

        MachineOperand &Reg = MI.getOperand(0);
        if (!Reg.isReg())
          continue;

        if (Reg.getReg() == 0)
          continue;

        const DIVariable *V = MI.getDebugVariable();
        const DIExpression *Exp = MI.getDebugExpression();
        write_dbg_val(Reg.getReg(), V, Exp, MRI, SIRI, OS);
      }
    }
  };

  json_array("debug_vals", WriteDebugVals, OS);
}

void write_function(MachineFunction &MF, LiveIntervals *LIS,
                    const MachineRegisterInfo &MRI, const SIInstrInfo *SIII,
                    const SIRegisterInfo *SIRI, raw_ostream &OS) {
  const SlotIndexes *SlotIndexes = LIS->getSlotIndexes();

  OS << "{\n";
  auto WriteName = [&MF, &OS]() { OS << MF.getName(); };
  json_pair("name", WriteName, OS);

  OS << ",\n";

  auto WriteBlocks = [&MF, &LIS, &MRI, &SIRI, &SIII, &OS]() {
    for (MachineBasicBlock &MBB : MF) {
      write_block(MBB, LIS, MRI, SIRI, SIII, OS);
      OS << ",\n";
    }
  };

  json_array("blocks", WriteBlocks, OS);

  OS << ",\n";

  auto WriteDefines = [&MF, &SlotIndexes, &MRI, &SIRI, &OS]() {
    write_defines(MF, SlotIndexes, MRI, SIRI, OS);
  };

  json_array("defines", WriteDefines, OS);

  OS << ",\n";

  auto WriteUses = [&MF, &SlotIndexes, &MRI, &SIRI, &OS]() {
    write_uses(MF, SlotIndexes, MRI, SIRI, OS);
  };

  json_array("uses", WriteUses, OS);

  OS << ",\n";

  auto WriteLiveness = [&MF, &LIS, &MRI, &SIRI, &OS]() {
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB) {
        if (MI.isDebugInstr())
          continue;
        const SlotIndex &SI = LIS->getInstructionIndex(MI).getBaseIndex();
        GCNRPTracker::LiveRegSet LISLR = llvm::getLiveRegs(SI, *LIS, MRI);
        write_liveness(SI, LISLR, MRI, SIRI, OS);
      }
  };

  json_array("liveness", WriteLiveness, OS);

  OS << ",\n";

  auto WriteLiveIntervals = [&MRI, &SIRI, &LIS, &OS]() {
    for (unsigned i = 0; i < MRI.getNumVirtRegs(); i++) {
      auto Reg = Register::index2VirtReg(i);
      if (!LIS->hasInterval(Reg))
        continue;
      auto &LI = LIS->getInterval(Reg);
      write_live_interval(LI, MRI, SIRI, OS);
    }
  };

  json_array("live_intervals", WriteLiveIntervals, OS);

  // Check debug info.
  const Function &F = MF.getFunction();
  const Module *M = F.getParent();
  const NamedMDNode *SourceMD = M->getNamedMetadata("dx.source.contents");
  if (SourceMD) {
    write_dbg_info(MF, LIS, MRI, SIII, SIRI, SlotIndexes, SourceMD, OS);
  }

  OS << "\n}";
}

void write_pressure(MachineFunction &MF, LiveIntervals *LIS,
                    const char *Filename) {
  int FD = -1;
  SmallString<128> TmpFilename(Filename);
  std::error_code EC = sys::fs::createUniqueFile(TmpFilename, FD, TmpFilename);
  if (EC) {
    errs() << "Error: " << EC.message() << "\n";
    return;
  }

  raw_fd_ostream O(FD, /*shouldClose=*/true);

  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const auto *SIII = ST->getInstrInfo();
  const auto *SIRI = ST->getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  write_function(MF, LIS, MRI, SIII, SIRI, O);
  O.flush();
  O.close();
}

void write_pressure(MachineFunction &MF, LiveIntervals *LIS, raw_ostream &OS) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const auto *SIII = ST->getInstrInfo();
  const auto *SIRI = ST->getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  write_function(MF, LIS, MRI, SIII, SIRI, OS);
  OS.flush();
}

} // namespace pressure
} // namespace llvm

namespace {
class ContributionList {
public:
  ContributionList(MachineFunction &MF) : MF(MF) {};
  void build();
  bool propagateContribution();
  MachineFunction &MF;
  DenseMap<MachineInstr *, unsigned> MIIndexMap;
  // Set of inst which contribute to build the key MachineInstr.
  DenseMap<MachineInstr *, DenseSet<MachineInstr *>> MIContributorMap;
  // Set of inst which been contributed by the key MachineInstr.
  DenseMap<MachineInstr *, DenseSet<MachineInstr *>> MIContributedToMap;
  void writeInst(MachineInstr &MI, const SIInstrInfo *SIII, raw_ostream &OS);
  void writeBlock(MachineBasicBlock &MBB, const SIInstrInfo *SIII,
                  raw_ostream &OS);
  void write(raw_ostream &OS);
};

void buildMIContribution(MachineInstr &MI,
                         DenseSet<MachineInstr *> &ContributorSet,
                         DenseSet<MachineInstr *> &ContributedSet,
                         MachineRegisterInfo &MRI) {
  for (MachineOperand &UseMO : MI.uses()) {
    if (!UseMO.isReg())
      continue;
    Register Reg = UseMO.getReg();
    if (Reg.isPhysical())
      continue;
    if (UseMO.isImplicit()) {
      // if (Reg == AMDGPU::EXEC || Reg == AMDGPU::EXEC_LO ||
      //    Reg == AMDGPU::SCC)
      continue;
    }
    for (MachineInstr &DefMI : MRI.def_instructions(Reg)) {
      ContributorSet.insert(&DefMI);
    }
  }

  for (MachineOperand &DstMO : MI.defs()) {
    if (!DstMO.isReg())
      continue;
    if (DstMO.isImplicit())
      continue;
    Register Reg = DstMO.getReg();
    if (Reg.isPhysical())
      continue;
    for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
      ContributedSet.insert(&UseMI);
    }
  }
}

bool ContributionList::propagateContribution() {
  bool IsUpdated = false;
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (auto *MBB : RPOT) {
    for (auto &MI : *MBB) {
      auto &Contributors = MIContributorMap[&MI];
      unsigned Size = Contributors.size();
      DenseSet<MachineInstr *> ParentContributors;
      for (auto *CMI : Contributors) {
        auto &Contributors = MIContributorMap[CMI];
        ParentContributors.insert(Contributors.begin(), Contributors.end());
      }
      Contributors.insert(ParentContributors.begin(), ParentContributors.end());
      IsUpdated |= Size < Contributors.size();
    }
  }
  return IsUpdated;
}

void ContributionList::build() {
  // Build contribution.
  auto &MRI = MF.getRegInfo();
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      auto &Contributors = MIContributorMap[&MI];
      auto &Contributed = MIContributedToMap[&MI];
      buildMIContribution(MI, Contributors, Contributed, MRI);
    }
  }
  // propagate contribution.
  bool IsUpdated = true;
  while (IsUpdated) {
    IsUpdated = propagateContribution();
  }
}

void ContributionList::writeInst(MachineInstr &MI, const SIInstrInfo *SIII,
                                 raw_ostream &OS) {
  OS << "\n{\n";
  unsigned ID = MIIndexMap[&MI];
  auto WriteSlot = [&ID, &OS]() { OS << ID; };

  json_pair("ID", WriteSlot, OS);

  OS << ",";

  auto WriteAsm = [&MI, &SIII, &OS]() {
    MI.print(OS, /*IsStandalone*/ true, /*SkipOpers*/ false,
             /*SkipDebugLoc*/ true, /*AddNewLine*/ false, SIII);
  };
  json_pair("asm", WriteAsm, OS);

  OS << ",\n";

  auto &Contributors = MIContributorMap[&MI];
  auto WriteContributor = [&Contributors, this, &OS]() {
    for (auto *MI : Contributors) {
      unsigned ID = MIIndexMap[MI];
      OS << ID << ",";
    }
  };

  json_array("contributors", WriteContributor, OS);
  OS << ",\n";

  auto &Contributeds = MIContributedToMap[&MI];
  auto WriteContributed = [&Contributeds, this, &OS]() {
    for (auto *MI : Contributeds) {
      unsigned ID = MIIndexMap[MI];
      OS << ID << ",";
    }
  };

  json_array("contributed", WriteContributed, OS);
  OS << "\n}\n";
}

void ContributionList::writeBlock(MachineBasicBlock &MBB,
                                  const SIInstrInfo *SIII, raw_ostream &OS) {
  OS << "{\n";
  auto WriteName = [&MBB, &OS]() { OS << MBB.getName(); };
  json_pair("name", WriteName, OS);

  OS << ",";

  auto WriteIndex = [&MBB, &OS]() { OS << MBB.getNumber(); };
  json_pair("id", WriteIndex, OS);

  OS << ",\n";

  auto WriteInsts = [this, &MBB, &SIII, &OS]() {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      writeInst(MI, SIII, OS);
      OS << ",\n";
    }
  };

  json_array("instructions", WriteInsts, OS);

  OS << ",\n";

  auto WritePreds = [&MBB, &OS]() {
    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      OS << Pred->getNumber() << ",";
    }
  };

  json_array("preds", WritePreds, OS);

  OS << ",";

  auto WriteSuccs = [&MBB, &OS]() {
    for (MachineBasicBlock *Succ : MBB.successors()) {
      OS << Succ->getNumber() << ",";
    }
  };

  json_array("succs", WriteSuccs, OS);

  OS << "}";
}

void ContributionList::write(raw_ostream &OS) {
  unsigned ID = 0;
  // Build ID for Write.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (auto *MBB : RPOT) {
    for (auto &MI : *MBB) {
      MIIndexMap[&MI] = ID++;
    }
  }

  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const auto *SIII = ST->getInstrInfo();

  OS << "{\n";
  auto WriteName = [this, &OS]() { OS << MF.getName(); };
  json_pair("name", WriteName, OS);

  OS << ",\n";

  auto WriteBlocks = [this, &SIII, &RPOT, &OS]() {
    for (auto *MBB : RPOT) {
      writeBlock(*MBB, SIII, OS);
      OS << ",\n";
    }
  };

  json_array("blocks", WriteBlocks, OS);

  OS << "\n}";
}
} // namespace

namespace llvm {

void write_contribution_list(llvm::MachineFunction &MF, const char *Filename) {
  int FD = -1;
  SmallString<128> TmpFilename(Filename);
  std::error_code EC = sys::fs::createUniqueFile(TmpFilename, FD, TmpFilename);
  if (EC) {
    errs() << "Error: " << EC.message() << "\n";
    return;
  }

  raw_fd_ostream O(FD, /*shouldClose=*/true);
  ContributionList CL(MF);
  CL.build();

  CL.write(O);

  O.flush();
  O.close();
}
} // namespace llvm

static bool isPhysReg(const MachineOperand &Op) {
  return Op.isReg() && Op.getReg().isPhysical();
}

// Sometimes split bb uses physical registers defined in BB, have to add them to
// live-in or the ir is malformed.
void llvm::updatePhysRegLiveInForBlock(MachineBasicBlock *NewBB,
                                       const MachineRegisterInfo *MRI) {
  // Initialize with current set of liveins. For new blocks this will be empty.
  SmallDenseSet<unsigned, 8> DefSet;
  for (const MachineBasicBlock::RegisterMaskPair &P : NewBB->liveins()) {
    DefSet.insert(P.PhysReg);
  }

  for (auto &MI : *NewBB) {
    // Add all undefined physical registers to the live in set.
    for (MachineOperand &Use : MI.operands()) {
      // Only process physreg uses.
      if (!isPhysReg(Use) || !Use.isUse())
        continue;

      // Reserved regs do not need to be tracked through live-in sets.
      Register Reg = Use.getReg();
      if (Use.isImplicit() && MRI && MRI->isReserved(Reg))
        continue;

      if (!DefSet.count(Reg))
        NewBB->addLiveIn(Reg);
    }

    // Add all physical register defs (exlicit+implicit) to the def register
    // set.
    for (MachineOperand &Def : MI.operands()) {
      // Only process physreg defs.
      if (!isPhysReg(Def) || !Def.isDef())
        continue;
      DefSet.insert(Def.getReg());
    }
  }
}

void llvm::buildPhysRegLiveInForBlock(MachineBasicBlock *NewBB,
                                      SmallDenseSet<unsigned, 8> &LiveOutSet,
                                      const MachineRegisterInfo *MRI) {
  for (auto RIt = NewBB->rbegin(); RIt != NewBB->rend(); RIt++) {
    auto &MI = *RIt;
    // Add all physical register defs (exlicit+implicit) to the def register
    // set.
    for (MachineOperand &Def : MI.operands()) {
      // Only process physreg defs.
      if (!isPhysReg(Def) || !Def.isDef())
        continue;
      LiveOutSet.erase(Def.getReg());
    }
    // Add all undefined physical registers to the live in set.
    for (MachineOperand &Use : MI.operands()) {
      // Only process physreg uses.
      if (!isPhysReg(Use) || !Use.isUse())
        continue;

      // Reserved regs do not need to be tracked through live-in sets.
      Register Reg = Use.getReg();
      if (Use.isImplicit() && MRI && MRI->isReserved(Reg))
        continue;

      if (!LiveOutSet.count(Reg))
        LiveOutSet.insert(Reg);
    }
  }
  for (unsigned Reg : LiveOutSet) {
    NewBB->addLiveIn(Reg);
  }
}

MachineReg llvm::createVirtualRegForOperand(MachineOpcode Opcode,
                                            unsigned OpNum,
                                            MachineFunction &MF) {
  const TargetSubtargetInfo &ST = MF.getSubtarget();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();
  const TargetInstrInfo *TII = ST.getInstrInfo();
  const MCInstrDesc &Desc = TII->get(Opcode);
  const TargetRegisterClass *RC = TII->getRegClass(Desc, OpNum, TRI, MF);
  if (!RC) {
    llvm::report_fatal_error(
        "Unable to create virtual reg for instruction operand");
  }

  MachineRegisterInfo &MRI = MF.getRegInfo();
  return MRI.createVirtualRegister(RC);
}

MachineReg llvm::createVirtualDstReg(MachineOpcode Opcode,
                                     MachineFunction &MF) {
  return llvm::createVirtualRegForOperand(Opcode, 0, MF);
}

// Return true if the MI is a copy of exec.
// If true then sets pDst to the destination register.
bool llvm::isExecCopy(const MachineInstr &MI, MachineReg Exec,
                      MachineReg *OutDst) {
  enum { DST = 0, SRC = 1 };
  bool FoundCopy = false;
  if (MI.getOpcode() == AMDGPU::COPY || MI.getOpcode() == AMDGPU::S_MOV_B32 ||
      MI.getOpcode() == AMDGPU::S_MOV_B64) {
    const MachineOperand &Src = MI.getOperand(SRC);
    if (Src.isReg() && Src.getReg() == Exec) {
      FoundCopy = true;
    }
  }
  if (FoundCopy) {
    *OutDst = MI.getOperand(DST).getReg();
  }

  return FoundCopy;
}

bool llvm::isSccLiveAt(llvm::MachineBasicBlock *MBB,
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
MachineBasicBlock::iterator llvm::findOrCreateInsertionPointForSccDef(
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

} // namespace

// In case like float4 v, v.x used and defined in one block, v.y used and define
// in another block, one live interval could touch more than one MBB.
// TouchedMBBSet is used for scheduling where local live interval could cross
// multiple regions, need to calculate livereg for each region inside touched
// MBB.
bool llvm::isLocalLiveInterval(
    const LiveInterval &LI, SlotIndexes *Indexes,
    SmallDenseSet<MachineBasicBlock *, 2> &TouchedMBBSet) {
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges()) {
      if (!isLocalLiveRange(&S, Indexes, TouchedMBBSet))
        return false;
    }
  }
  return isLocalLiveRange(&LI, Indexes, TouchedMBBSet);
}

bool llvm::isLocalLiveInterval(const LiveInterval &LI, SlotIndexes *Indexes) {
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges()) {
      if (!isLocalLiveRange(&S, Indexes))
        return false;
    }
  }
  return isLocalLiveRange(&LI, Indexes);
}

// This is used to speed up reg pressure calculation.
// If instruction is moved, the cached liveset will be out of date.
// Before instruction is moved, the value will be correct.
void llvm::buildEndLiveMap(
    llvm::LiveIntervals *LIS, llvm::MachineFunction &MF,
    const llvm::MachineRegisterInfo &MRI,
    llvm::DenseMap<llvm::MachineBasicBlock *, LiveSet> &MBBLiveMap,
    bool After) {
  // When only have one block, end live reg must be empty.
  if (MF.size() == 1)
    return;
  auto *SlotIndexes = LIS->getSlotIndexes();
  DenseMap<MachineBasicBlock *, SlotIndex> MBBOutputSlotMap;
  for (MachineBasicBlock &MBB : MF) {
    auto BBEnd = MBB.rbegin();

    // R.End doesn't point to the boundary instruction.
    // Skip Debug instr.
    if (llvm::getNonDebugMBBEnd(BBEnd, MBB)) {
      auto SI = SlotIndexes->getInstructionIndex(*BBEnd);
      MBBOutputSlotMap[&MBB] = After ? SI.getDeadSlot() : SI.getBaseIndex();
    }
  }

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;

    const auto &LI = LIS->getInterval(Reg);

    // Skip local live interval to make live input/ouput faster.
    if (llvm::isLocalLiveInterval(LI, SlotIndexes))
      continue;

    for (auto OutputIt : MBBOutputSlotMap) {
      MachineBasicBlock *MBB = OutputIt.first;
      auto SI = OutputIt.second;

      auto LiveMask = getLiveLaneMask(Reg, SI, *LIS, MRI);
      if (LiveMask.any())
        MBBLiveMap[MBB][Reg] = LiveMask;
    }
  }
}

unsigned llvm::getCurrentVGPRCount(llvm::MachineFunction &MF,
                                   const SIRegisterInfo *SIRI) {
  auto &MRI = MF.getRegInfo();
  for (MCPhysReg Reg : reverse(AMDGPU::VGPR_32RegClass.getRegisters())) {
    if (MRI.isPhysRegUsed(Reg)) {
      return SIRI->getHWRegIndex(Reg) - SIRI->getHWRegIndex(AMDGPU::VGPR0) + 1;
    }
  }
  return 0;
}

unsigned llvm::getCurrentSGPRCount(llvm::MachineFunction &MF,
                                   const SIRegisterInfo *SIRI) {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  Register ScratchRSrcReg = MFI->getScratchRSrcReg();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned MaxSGPR = 0;
  for (MCPhysReg Reg : reverse(AMDGPU::SGPR_32RegClass.getRegisters())) {
    if (MRI.isPhysRegUsed(Reg)) {
      // Skip scratch reserved reg, which is a big register that don't really
      // contribute to this stat.
      if (ScratchRSrcReg != 0) {
        if (SIRI->isSubRegister(ScratchRSrcReg, Reg))
          continue;
      }
      MaxSGPR = SIRI->getHWRegIndex(Reg) - SIRI->getHWRegIndex(AMDGPU::SGPR0);
      break;
    }
  }
  return 1 + llvm::RegForVCC + MaxSGPR;
}

void llvm::dumpLiveSet(const LiveSet &LiveSet, const SIRegisterInfo *SIRI) {

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

// Test if all fast math flags of this Machine Instr are set. This allows
// all non-strict floating-point transforms.
bool llvm::isFastMathInst(llvm::MachineInstr &MI) {
  // Follow the checks in isFast() in SelectionDAGNodes.h
  return MI.getFlag(llvm::MachineInstr::MIFlag::FmNsz) &&
         MI.getFlag(llvm::MachineInstr::MIFlag::FmArcp) &&
         MI.getFlag(llvm::MachineInstr::MIFlag::FmNoNans) &&
         MI.getFlag(llvm::MachineInstr::MIFlag::FmNoInfs) &&
         MI.getFlag(llvm::MachineInstr::MIFlag::FmContract) &&
         MI.getFlag(llvm::MachineInstr::MIFlag::FmAfn) &&
         MI.getFlag(llvm::MachineInstr::MIFlag::FmReassoc);
}
#if 0
bool llvm::IsLdsSpillSupportedForHwStage(xmd::HwStage Stage)
{
    switch (Stage)
    {
    case xmd::HwStage::PS:
    case xmd::HwStage::CS:
        return true;
    default:
        return false;
    }
}
#endif

MachineBasicBlock::succ_iterator
llvm::findSuccessor(llvm::MachineBasicBlock *MBB,
                    llvm::MachineBasicBlock *Succ) {
  for (MachineBasicBlock::succ_iterator It = MBB->succ_begin(),
                                        End = MBB->succ_end();
       It != End; ++It) {
    if (*It == Succ) {
      return It;
    }
  }

  return MBB->succ_end();
}
