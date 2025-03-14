#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/SlotIndexes.h"

// #include "dxc/DXIL/DxilMetadataHelper.h"
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
      auto &phiInsts = blockToPhiInstsMap[&BB];
      for (MachineInstr &I : BB) {
        if (!I.isPHI())
          break;
        phiInsts.insert(&I);
        unsigned Reg = I.getOperand(0).getReg();
        // Add incoming values.
        for (unsigned i = 1; i < I.getNumOperands(); i += 2) {
          MachineOperand &MO = I.getOperand(i);
          if (!MO.isReg())
            continue;
          MachineInstr *DefMI = MRI.getUniqueVRegDef(MO.getReg());
          if (!DefMI)
            continue;
          blockToPhiInstsMap[DefMI->getParent()].insert(DefMI);
        }
        // Add users.
        for (MachineInstr &UseMI : MRI.use_nodbg_instructions(Reg)) {
          blockToPhiInstsMap[UseMI.getParent()].insert(&UseMI);
        }
      }
    }
  } /// Adds custom features for a visualization of the ScheduleDAG.
  void addCustomGraphFeatures(llvm::GraphWriter<CFGWithPhi *> &) const {}
  MachineFunction &F;
  DenseMap<const MachineBasicBlock *, DenseSet<MachineInstr *>>
      blockToPhiInstsMap;
  void dump();
};

void CFGWithPhi::dump() {
#ifdef DBG
  for (MachineBasicBlock &BB : F) {
    dbgs() << BB.getName() << "\n";
    auto &phiInsts = blockToPhiInstsMap[&BB];
    for (MachineInstr *I : phiInsts) {
      if (!I->isPHI())
        continue;
      I->dump();
    }
    for (MachineInstr *I : phiInsts) {
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

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(const CFGWithPhi *G) {
    return "CFG with Phi graph";
  }

  static std::string getNodeIdentifierLabel(const MachineBasicBlock *Node,
                                            const CFGWithPhi *Graph) {
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
    auto it = G->blockToPhiInstsMap.find(BB);
    if (it != G->blockToPhiInstsMap.end()) {

      auto &phiInsts = it->second;
      for (MachineInstr *I : phiInsts) {
        if (!I->isPHI())
          continue;
        I->print(OS);
        OS << "\n";
      }
      for (MachineInstr *I : phiInsts) {
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
                                        const CFGWithPhi *G) {
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
  LaneBitmask mask = Mask;
  if (mask.any()) {
    if (unsigned maskSize = mask.getNumLanes()) {
      if (maskSize < Size)
        Size = maskSize;
    }
  }
  return Size;
}

void CollectLiveSetPressure(const LiveSet &liveSet,
                            const MachineRegisterInfo &MRI,
                            const SIRegisterInfo *SIRI, unsigned &VPressure,
                            unsigned &SPressure) {
  VPressure = 0;
  SPressure = 0;
  for (auto liveIt : liveSet) {
    unsigned Reg = liveIt.first;
    unsigned Size = getRegSize(Reg, liveIt.second, MRI, SIRI);
    if (SIRI->isVGPR(MRI, Reg)) {
      VPressure += Size;
    } else {
      SPressure += Size;
    }
  }
}

bool isExecUpdateForControlFlow(llvm::MachineInstr &MI) {
  bool isExecUpdate = false;
  unsigned opcode = MI.getOpcode();
  if (opcode == AMDGPU::S_MOV_B64 || opcode == AMDGPU::S_MOV_B32 ||
      opcode == AMDGPU::S_OR_B64_term || opcode == AMDGPU::S_OR_B32_term ||
      opcode == AMDGPU::S_OR_SAVEEXEC_B64 ||
      opcode == AMDGPU::S_OR_SAVEEXEC_B32 || opcode == AMDGPU::S_AND_B64 ||
      opcode == AMDGPU::S_AND_B32 || opcode == AMDGPU::S_ANDN2_B64 ||
      opcode == AMDGPU::S_ANDN2_B32) {
    MachineOperand &Dst = MI.getOperand(0);
    if (Dst.getReg() == AMDGPU::EXEC || Dst.getReg() == AMDGPU::EXEC_LO) {
      isExecUpdate = true;
    }
  }
  return isExecUpdate;
}

bool IsSub0Sub1SingleDef(unsigned Reg, const MachineRegisterInfo &MRI) {
  // Support multi def for pattern of pointer:
  // undef %808.sub0:sgpr_64 = COPY killed %795:sgpr_32
  // %808.sub1:sgpr_64 = S_MOV_B32 0
  bool bHasSub0 = false;
  bool bHasSub1 = false;
  for (MachineOperand &UserDefMO : MRI.def_operands(Reg)) {
    if (unsigned SubReg = UserDefMO.getSubReg()) {
      bool bSingleSubReg = false;
      switch (SubReg) {
      default:
        break;
      case AMDGPU::sub0:
        if (!bHasSub0) {
          bHasSub0 = true;
          bSingleSubReg = true;
        }
        break;
      case AMDGPU::sub1:
        if (!bHasSub1) {
          bHasSub1 = true;
          bSingleSubReg = true;
        }
        break;
      }
      if (!bSingleSubReg) {
        bHasSub0 = false;
        break;
      }
    } else {
      bHasSub0 = false;
      break;
    }
  }

  return (bHasSub0 && bHasSub1);
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

void mergeLiveRegSet(LiveSet &targetSet, const LiveSet &inputSet) {
  for (auto Reg : inputSet) {
    unsigned reg = Reg.first;
    LaneBitmask mask = Reg.second;
    auto targetReg = targetSet.find(reg);
    if (targetReg != targetSet.end()) {
      LaneBitmask targetMask = targetReg->second;
      mask |= targetMask;
    }
    targetSet[reg] = mask;
  }
}

void andLiveRegSet(LiveSet &targetSet, const LiveSet &inputSet) {
  GCNRPTracker::LiveRegSet AndSet;
  for (auto Reg : inputSet) {
    unsigned reg = Reg.first;
    LaneBitmask mask = Reg.second;
    auto targetReg = targetSet.find(reg);
    if (targetReg != targetSet.end()) {
      LaneBitmask targetMask = targetReg->second;
      mask &= targetMask;
      AndSet[reg] = mask;
    }
  }

  targetSet = AndSet;
}

void andNotLiveRegSet(LiveSet &targetSet, const LiveSet &inputSet) {
  for (auto Reg : inputSet) {
    unsigned reg = Reg.first;
    LaneBitmask mask = Reg.second;
    auto targetReg = targetSet.find(reg);
    if (targetReg != targetSet.end()) {
      LaneBitmask targetMask = targetReg->second;
      if ((targetMask | mask) == mask)
        targetSet.erase(reg);
      else
        targetSet[reg] = targetMask & (~mask);
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
  unsigned offset;
  unsigned size;
  static SmallVector<Piece, 8> split(std::bitset<32> mask) {

    SmallVector<Piece, 8> pieces;
    Piece piece = {0, 0, 0};
    for (unsigned i = 0; i < 32; i++) {
      if (mask.test(i)) {
        if (piece.size == 0)
          piece.offset = i;

        piece.size++;
        // Make sure no piece bigger than 8.
        if (piece.size == 8) {
          pieces.emplace_back(piece);
          piece.size = 0;
        }
      } else {
        if (piece.size == 0) {
          continue;
        }
        pieces.emplace_back(piece);
        piece.size = 0;
      }
    }
    return pieces;
  }
};

void updateSubReg(MachineOperand &UseMO, const llvm::TargetRegisterClass *NewRC,
                  unsigned offset, const SIRegisterInfo *SIRI,
                  const SIInstrInfo *SIII) {
  unsigned size = NewRC->getLaneMask().getNumLanes();
  if (size == 1) {
    UseMO.setSubReg(0);
  } else {
    const uint32_t SubReg = UseMO.getSubReg();
    LaneBitmask Mask = SIRI->getSubRegIndexLaneMask(SubReg);

    unsigned mask = Mask.getAsInteger() >> offset;

    unsigned NewSubReg = SIRI->getMinimalSpanningSubRegIdxSetForLaneMask(
                                 NewRC, LaneBitmask(mask))
                             .front();

    UseMO.setSubReg(NewSubReg);
  }
}

bool reduceChannel(unsigned offset, MachineInstr &MI, const MCInstrDesc &desc,
                   MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                   const SIInstrInfo *SIII, SlotIndexes *SlotIndexes) {
  MachineOperand &DstMO = MI.getOperand(0);
  // Skip case when dst subReg not 0.
  if (DstMO.getSubReg()) {
    return false;
  }
  unsigned Reg = DstMO.getReg();

  SmallVector<MachineOperand *, 2> UseMOs;
  for (MachineOperand &UseMO : MRI.use_nodbg_operands(Reg)) {
    UseMOs.emplace_back(&UseMO);
  }

  const llvm::TargetRegisterClass *NewRC =
      SIRI->getRegClass(desc.operands().front().RegClass);
  unsigned size = NewRC->getLaneMask().getNumLanes();
  if (offset > 0) {
    // Update offset operand in MI.
    MachineOperand *OffsetOp =
        SIII->getNamedOperand(MI, AMDGPU::OpName::offset);

    const uint32_t LaneSize = sizeof(uint32_t);
    if (OffsetOp) {
      if (OffsetOp->isImm()) {
        assert(OffsetOp != nullptr);
        int64_t Offset = OffsetOp->getImm();
        Offset += offset * LaneSize;
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
        unsigned NewOffsetReg =
            MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
        auto OffsetAdd = BuildMI(*MI.getParent()->getParent(), MI.getDebugLoc(),
                                 SIII->get(AMDGPU::S_ADD_U32))
                             .addDef(NewOffsetReg)
                             .add(*OffsetOp)
                             .addImm(offset * LaneSize);
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
      updateSubReg(*UseMO, NewRC, offset, SIRI, SIII);
    }
  } else if (size == 1) {
    // Clear subReg when size is 1.
    for (MachineOperand *UseMO : UseMOs) {
      UseMO->setSubReg(0);
    }
  }

  MI.setDesc(desc);
  // Mutate reg class of Reg.
  MRI.setRegClass(Reg, NewRC);
  return true;
}

bool removeUnusedLanes(llvm::MachineInstr &MI, MachineRegisterInfo &MRI,
                       const SIRegisterInfo *SIRI, const SIInstrInfo *SIII,
                       SlotIndexes *SlotIndexes) {
  bool bImm = false;
  switch (MI.getOpcode()) {
  default:
    break;
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM:
  case AMDGPU::S_BUFFER_LOAD_DWORDX16_IMM:
    bImm = true;
  case AMDGPU::S_BUFFER_LOAD_DWORDX2_SGPR:
  case AMDGPU::S_BUFFER_LOAD_DWORDX4_SGPR:
  case AMDGPU::S_BUFFER_LOAD_DWORDX8_SGPR:
  case AMDGPU::S_BUFFER_LOAD_DWORDX16_SGPR: {
    unsigned Reg = MI.getOperand(0).getReg();
    if (!MRI.getUniqueVRegDef(Reg))
      return false;
    LaneBitmask dstMask = getRegMask(MI.getOperand(0), MRI);
    LaneBitmask UseMask;
    for (MachineOperand &MO : MRI.use_operands(Reg)) {
      UseMask |= llvm::getRegMask(MO, MRI);
    }

    const unsigned fullMask = dstMask.getAsInteger();
    unsigned mask = UseMask.getAsInteger();
    if (mask == fullMask)
      return false;
    // Split mask when there's gap. Then group mask to 2/4/8.
    auto pieces = Piece::split(std::bitset<32>(mask));
    // Now only support 1 piece.
    if (pieces.size() != 1)
      return false;
    auto piece = pieces[0];
    if (piece.size > 8)
      return false;

    // TODO: enable offset support when bImm is true.
    // Now if break different test when mul LaneSize or not mul for the offset.
    if (bImm && piece.offset != 0)
      return false;

    switch (piece.size) {
    default:
      return false;
    case 1:
      return reduceChannel(piece.offset, MI,
                           SIII->get(bImm ? AMDGPU::S_BUFFER_LOAD_DWORD_IMM
                                          : AMDGPU::S_BUFFER_LOAD_DWORD_SGPR),
                           MRI, SIRI, SIII, SlotIndexes);
    case 2:
      return reduceChannel(piece.offset, MI,
                           SIII->get(bImm ? AMDGPU::S_BUFFER_LOAD_DWORDX2_IMM
                                          : AMDGPU::S_BUFFER_LOAD_DWORDX2_SGPR),
                           MRI, SIRI, SIII, SlotIndexes);
    case 3:
      if (fullMask == 0xf)
        return false;
    case 4:
      return reduceChannel(piece.offset, MI,
                           SIII->get(bImm ? AMDGPU::S_BUFFER_LOAD_DWORDX4_IMM
                                          : AMDGPU::S_BUFFER_LOAD_DWORDX4_SGPR),
                           MRI, SIRI, SIII, SlotIndexes);
    case 5:
    case 6:
    case 7:
      if (fullMask == 0xff)
        return false;
    case 8:
      return reduceChannel(piece.offset, MI,
                           SIII->get(bImm ? AMDGPU::S_BUFFER_LOAD_DWORDX8_IMM
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
                  DenseSet<MachineBasicBlock *> &hotMBBs) {
  bool bCross = false;
  for (MachineBasicBlock *hotBB : hotMBBs) {
    if (reach_block(BB, DT, PDT, LI, hotBB)) {
      bCross = true;
      break;
    }
  }
  return bCross;
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
bool GetNonDebugMBBEnd(MachineBasicBlock::reverse_iterator &BBEnd,
                       MachineBasicBlock &MBB) {
  // R.End doesn't point to the boundary instruction.
  // Skip Debug instr.
  while (BBEnd != MBB.rend() && BBEnd->isDebugInstr())
    BBEnd++;
  return BBEnd != MBB.rend();
}
} // namespace llvm

// Helper functions to write jason.
namespace {
void json_name(StringRef Val, raw_ostream &os) { os << "\"" << Val << "\":"; }

template <typename write_fn>
void json_pair(StringRef Val, write_fn &fn, raw_ostream &os) {
  json_name(Val, os);
  os << "\"";
  fn();
  os << "\"";
}

template <typename write_fn>
void json_obj_pair(StringRef Val, write_fn &fn, raw_ostream &os) {
  json_name(Val, os);

  fn();
}

template <typename write_fn>
void json_array(StringRef Val, write_fn &fn, raw_ostream &os) {
  json_name(Val, os);
  os << "[";
  fn();
  os << "]";
}
} // namespace

namespace llvm {
namespace pressure {

void write_inst(MachineInstr &MI, const SlotIndexes *SlotIndexes,
                const SIInstrInfo *SIII, raw_ostream &os) {
  os << "{";
  SlotIndex Slot = SlotIndexes->getInstructionIndex(MI);
  auto writeSlot = [&Slot, &os]() { Slot.print(os); };

  json_pair("slot_index", writeSlot, os);

  os << ",";

  auto writeOpcode = [&MI, &SIII, &os]() {
    os << SIII->getName(MI.getOpcode());
  };

  json_pair("opcode", writeOpcode, os);

  os << ",";

  auto writeAsm = [&MI, &SIII, &os]() {
    MI.print(os, /*IsStandalone*/ true, /*SkipOpers*/ false,
             /*SkipDebugLoc*/ true, /*AddNewLine*/ false, SIII);
  };
  json_pair("asm", writeAsm, os);

  os << "}";
}

void print_reg(Register Reg, const MachineRegisterInfo &MRI,
               const SIRegisterInfo *SIRI, raw_ostream &os) {
  if (Reg.isVirtual()) {
    StringRef Name = MRI.getVRegName(Reg);
    if (Name != "") {
      os << '%' << Name;
    } else {
      os << '%' << Register::virtReg2Index(Reg);
    }
  } else if (Reg < SIRI->getNumRegs()) {
    os << '$';
    printLowerCase(SIRI->getName(Reg), os);
  } else {
    llvm_unreachable("invalid reg");
  }
}

void write_reg(unsigned Reg, unsigned SubReg, const MachineRegisterInfo &MRI,
               const SIRegisterInfo *SIRI, raw_ostream &os) {
  os << "{";

  auto writeReg = [&MRI, &SIRI, &Reg, &os]() { print_reg(Reg, MRI, SIRI, os); };
  json_pair("reg", writeReg, os);

  os << ",";

  auto writeSubReg = [&SubReg, &os]() { os << SubReg; };

  json_pair("sub_reg", writeSubReg, os);

  os << ",";
  auto writeIsSgpr = [&Reg, &MRI, &SIRI, &os]() {
    if (SIRI->isSGPRReg(MRI, Reg))
      os << "true";
    else
      os << "false";
  };
  json_obj_pair("is_sgpr", writeIsSgpr, os);
  os << "}";
}

unsigned get_reg_size(unsigned Reg, const MachineRegisterInfo &MRI,
                      const SIRegisterInfo *SIRI) {
  return SIRI->getRegClassForReg(MRI, Reg)->getLaneMask().getNumLanes();
}

void write_live(unsigned Reg, LaneBitmask Mask, const MachineRegisterInfo &MRI,
                const SIRegisterInfo *SIRI, raw_ostream &os) {
  if (Mask.none()) {
    unsigned size = get_reg_size(Reg, MRI, SIRI);
    Mask = LaneBitmask((1 << size) - 1);
  }
  unsigned mask = Mask.getAsInteger();
  for (unsigned i = 0; i <= Mask.getHighestLane(); i++) {
    if (mask & (1 << i)) {
      write_reg(Reg, i, MRI, SIRI, os);
      os << ",\n";
    }
  }
}

void write_dag_input_node(unsigned ID, unsigned reg, unsigned mask,
                          const MachineRegisterInfo &MRI,
                          const SIRegisterInfo *SIRI, raw_ostream &os) {
  os << "{";
  auto writeID = [&ID, &os]() { os << ID; };

  json_pair("ID", writeID, os);

  os << ",";

  auto writeReg = [&reg, &MRI, &SIRI, &os]() { print_reg(reg, MRI, SIRI, os); };

  json_pair("reg", writeReg, os);

  os << ",";

  auto writeMask = [&mask, &os]() { os << mask; };

  json_pair("mask", writeMask, os);

  os << "},\n";
}

void write_dag_inst_node(unsigned ID, SlotIndex Slot,
                         GCNRPTracker::LiveRegSet LiveReg,
                         const MachineRegisterInfo &MRI,
                         const SIRegisterInfo *SIRI, SUnit *SU,
                         raw_ostream &os) {
  os << "{";
  auto writeID = [&ID, &os]() { os << ID; };

  json_pair("ID", writeID, os);

  os << ",";

  auto writeSlot = [&Slot, &os]() { Slot.print(os); };

  json_pair("slot_index", writeSlot, os);

  os << ",";

  auto writeRegs = [&LiveReg, &MRI, &SIRI, &os]() {
    for (auto it : LiveReg) {
      unsigned Reg = it.first;
      LaneBitmask Mask = it.second;
      write_live(Reg, Mask, MRI, SIRI, os);
    }
  };
  json_array("regs", writeRegs, os);

  os << ",";

  auto writePreds = [&SU, &os]() {
    for (auto &Pred : SU->Preds) {

      os << Pred.getSUnit()->NodeNum << ",";
    }
  };

  json_array("preds", writePreds, os);

  os << "},\n";
}

void write_block(MachineBasicBlock &Blk, LiveIntervals *LIS,
                 const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                 const SIInstrInfo *SIII, raw_ostream &os) {
  os << "{\n";
  auto writeName = [&Blk, &os]() { os << Blk.getName(); };
  json_pair("name", writeName, os);

  os << ",";

  auto writeIndex = [&Blk, &os]() { os << Blk.getNumber(); };
  json_pair("id", writeIndex, os);

  os << ",";

  const SlotIndexes *SlotIndexes = LIS->getSlotIndexes();

  SlotIndex BeginSlot = SlotIndexes->getMBBStartIdx(&Blk);
  auto writeSlot = [&BeginSlot, &os]() { BeginSlot.print(os); };
  json_pair("begin_slot", writeSlot, os);

  os << ",";

  SlotIndex EndSlot = SlotIndexes->getMBBEndIdx(&Blk);
  auto writeEndSlot = [&EndSlot, &os]() { EndSlot.print(os); };
  json_pair("end_slot", writeEndSlot, os);

  os << ",";

  auto writeInsts = [&Blk, &SlotIndexes, &SIII, &os]() {
    for (MachineInstr &MI : Blk) {
      if (MI.isDebugInstr())
        continue;
      write_inst(MI, SlotIndexes, SIII, os);
      os << ",\n";
    }
  };

  json_array("instructions", writeInsts, os);

  os << ",";

  BlockExpDag dag(&Blk, LIS, MRI, SIRI, SIII);
  dag.buildWithPressure();

  const auto StartLiveReg = llvm::getLiveRegs(BeginSlot, *dag.LIS, dag.MRI);
  auto writeInputs = [&StartLiveReg, &dag, &os]() {
    for (auto it : StartLiveReg) {
      unsigned Reg = it.first;
      LaneBitmask mask = it.second;
      SUnit *SU = dag.InputSUnitMap[Reg];
      // Write Reg and mask to the nodes.
      write_dag_input_node(SU->NodeNum, Reg, mask.getAsInteger(), dag.MRI,
                           dag.SIRI, os);
    }
  };

  json_array("input_nodes", writeInputs, os);

  os << ",";

  auto writeNodes = [&SlotIndexes, &dag, &os]() {
    for (auto it : dag.MISUnitMap) {
      MachineInstr *MI = it.first;
      SUnit *SU = it.second;
      // Use SlotIndex of MI.
      SlotIndex SlotIndex;
      if (!MI->isDebugInstr())
        SlotIndex = SlotIndexes->getInstructionIndex(*MI);
      GCNRPTracker::LiveRegSet LiveReg = dag.DagPressureMap[SU];
      // Write slot, live to the nodes.
      write_dag_inst_node(SU->NodeNum, SlotIndex, LiveReg, dag.MRI, dag.SIRI,
                          SU, os);
    }
  };

  json_array("inst_nodes", writeNodes, os);

  os << ",";

  auto writePreds = [&Blk, &os]() {
    for (MachineBasicBlock *Pred : Blk.predecessors()) {
      os << Pred->getNumber() << ",";
    }
  };

  json_array("preds", writePreds, os);

  os << ",";

  auto writeSuccs = [&Blk, &os]() {
    for (MachineBasicBlock *Succ : Blk.successors()) {
      os << Succ->getNumber() << ",";
    }
  };

  json_array("succs", writeSuccs, os);

  os << "}";
}

void write_define(SlotIndex &Slot, unsigned Reg, unsigned SubReg,
                  const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                  raw_ostream &os) {
  os << "{";
  auto writeSlot = [&Slot, &os]() { Slot.print(os); };

  json_pair("slot_index", writeSlot, os);

  os << ",";

  auto writeReg = [&MRI, &SIRI, &Reg, &SubReg, &os]() {
    write_reg(Reg, SubReg, MRI, SIRI, os);
  };
  json_obj_pair("reg", writeReg, os);

  os << "}\n";

  os << ",";
}

void write_define(MachineOperand &MO, const SlotIndexes *SlotIndexes,
                  const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                  raw_ostream &os) {
  // Split subReg?  MO.getSubReg();
  unsigned Reg = MO.getReg();
  unsigned SubReg = MO.getSubReg();
  MachineInstr *MI = MO.getParent();
  SlotIndex Slot = SlotIndexes->getInstructionIndex(*MI);
  if (SubReg == 0) {
    unsigned size = get_reg_size(Reg, MRI, SIRI);
    for (unsigned i = 0; i < size; i++) {
      write_define(Slot, Reg, i, MRI, SIRI, os);
    }
  } else {
    switch (SubReg) {
    default:
      assert(0 && "SubReg not supported yet.");
      write_define(Slot, Reg, SubReg, MRI, SIRI, os);
      break;
    case AMDGPU::sub0:
      write_define(Slot, Reg, 0, MRI, SIRI, os);
      break;
    case AMDGPU::sub1:
      write_define(Slot, Reg, 1, MRI, SIRI, os);
      break;
    case AMDGPU::sub2:
      write_define(Slot, Reg, 2, MRI, SIRI, os);
      break;
    case AMDGPU::sub3:
      write_define(Slot, Reg, 3, MRI, SIRI, os);
      break;
    case AMDGPU::sub4:
      write_define(Slot, Reg, 4, MRI, SIRI, os);
      break;
    case AMDGPU::sub5:
      write_define(Slot, Reg, 5, MRI, SIRI, os);
      break;
    case AMDGPU::sub6:
      write_define(Slot, Reg, 6, MRI, SIRI, os);
      break;
    case AMDGPU::sub7:
      write_define(Slot, Reg, 7, MRI, SIRI, os);
      break;
    case AMDGPU::sub8:
      write_define(Slot, Reg, 8, MRI, SIRI, os);
      break;
    case AMDGPU::sub9:
      write_define(Slot, Reg, 9, MRI, SIRI, os);
      break;
    case AMDGPU::sub10:
      write_define(Slot, Reg, 10, MRI, SIRI, os);
      break;
    case AMDGPU::sub11:
      write_define(Slot, Reg, 11, MRI, SIRI, os);
      break;
    case AMDGPU::sub12:
      write_define(Slot, Reg, 12, MRI, SIRI, os);
      break;
    case AMDGPU::sub13:
      write_define(Slot, Reg, 13, MRI, SIRI, os);
      break;
    case AMDGPU::sub14:
      write_define(Slot, Reg, 14, MRI, SIRI, os);
      break;
    case AMDGPU::sub15:
      write_define(Slot, Reg, 15, MRI, SIRI, os);
      break;
    case AMDGPU::sub0_sub1:
      write_define(Slot, Reg, 0, MRI, SIRI, os);
      write_define(Slot, Reg, 1, MRI, SIRI, os);
      break;
    case AMDGPU::sub2_sub3:
      write_define(Slot, Reg, 2, MRI, SIRI, os);
      write_define(Slot, Reg, 3, MRI, SIRI, os);
      break;
    case AMDGPU::sub4_sub5:
      write_define(Slot, Reg, 4, MRI, SIRI, os);
      write_define(Slot, Reg, 5, MRI, SIRI, os);
      break;
    case AMDGPU::sub1_sub2:
      write_define(Slot, Reg, 1, MRI, SIRI, os);
      write_define(Slot, Reg, 2, MRI, SIRI, os);
      break;
    case AMDGPU::sub0_sub1_sub2:
      write_define(Slot, Reg, 0, MRI, SIRI, os);
      write_define(Slot, Reg, 1, MRI, SIRI, os);
      write_define(Slot, Reg, 2, MRI, SIRI, os);
      break;
    case AMDGPU::sub0_sub1_sub2_sub3:
      write_define(Slot, Reg, 0, MRI, SIRI, os);
      write_define(Slot, Reg, 1, MRI, SIRI, os);
      write_define(Slot, Reg, 2, MRI, SIRI, os);
      write_define(Slot, Reg, 3, MRI, SIRI, os);
      break;
    case AMDGPU::sub2_sub3_sub4_sub5:
      write_define(Slot, Reg, 2, MRI, SIRI, os);
      write_define(Slot, Reg, 3, MRI, SIRI, os);
      write_define(Slot, Reg, 4, MRI, SIRI, os);
      write_define(Slot, Reg, 5, MRI, SIRI, os);
      break;
    case AMDGPU::sub0_sub1_sub2_sub3_sub4_sub5_sub6_sub7:
      write_define(Slot, Reg, 0, MRI, SIRI, os);
      write_define(Slot, Reg, 1, MRI, SIRI, os);
      write_define(Slot, Reg, 2, MRI, SIRI, os);
      write_define(Slot, Reg, 3, MRI, SIRI, os);
      write_define(Slot, Reg, 4, MRI, SIRI, os);
      write_define(Slot, Reg, 5, MRI, SIRI, os);
      write_define(Slot, Reg, 6, MRI, SIRI, os);
      write_define(Slot, Reg, 7, MRI, SIRI, os);
      break;
    }
  }
}

void write_defines(MachineFunction &MF, const SlotIndexes *SlotIndexes,
                   const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                   raw_ostream &os) {

  for (unsigned i = 0; i < MRI.getNumVirtRegs(); i++) {
    auto Reg = Register::index2VirtReg(i);

    for (MachineOperand &MO : MRI.def_operands(Reg)) {
      write_define(MO, SlotIndexes, MRI, SIRI, os);
    }
  }
}

void write_uses(MachineFunction &MF, const SlotIndexes *SlotIndexes,

                const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                raw_ostream &os) {

  for (unsigned i = 0; i < MRI.getNumVirtRegs(); i++) {
    auto Reg = Register::index2VirtReg(i);

    for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
      // TODO: create write_use if use has more info.
      write_define(MO, SlotIndexes, MRI, SIRI, os);
    }
  }
}

void write_liveness(SlotIndex Slot, GCNRPTracker::LiveRegSet &LiveSet,
                    const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                    raw_ostream &os) {
  os << "{";
  auto writeSlot = [&Slot, &os]() { Slot.print(os); };

  json_pair("slot_index", writeSlot, os);

  os << ",";

  auto writeRegs = [&LiveSet, &MRI, &SIRI, &os]() {
    for (auto it : LiveSet) {
      unsigned Reg = it.first;
      LaneBitmask Mask = it.second;
      write_live(Reg, Mask, MRI, SIRI, os);
    }
  };
  json_array("regs", writeRegs, os);
  os << "\n},\n";
}

void write_segment(const LiveInterval::Segment &S, raw_ostream &os) {
  os << "{";
  auto writeBegin = [&S, &os]() { S.start.print(os); };

  json_pair("begin", writeBegin, os);

  os << ",";

  auto writeEnd = [&S, &os]() { S.end.print(os); };

  json_pair("end", writeEnd, os);

  os << ",";

  auto writeValNum = [&S, &os]() {
    if (S.valno)
      os << S.valno->id;
    else
      os << 0xFFFFFFFF;
  };

  json_pair("val_num", writeValNum, os);

  os << "},\n";
}

void write_subrange(const LiveInterval::SubRange &SR, raw_ostream &os) {
  os << "{\n";
  auto writeMask = [&SR, &os]() { os << SR.LaneMask.getAsInteger(); };

  json_pair("mask", writeMask, os);

  os << ",";

  // Segments.
  auto writeSegments = [&SR, &os]() {
    for (auto &S : SR.segments) {
      write_segment(S, os);
    }
  };

  json_array("segments", writeSegments, os);

  os << "\n},\n";
}

void write_live_interval(LiveInterval &LI, const MachineRegisterInfo &MRI,
                         const SIRegisterInfo *SIRI, raw_ostream &os) {
  os << "{\n";

  auto writeReg = [&LI, &MRI, &SIRI, &os]() {
    write_reg(LI.reg(), 0, MRI, SIRI, os);
  };

  json_obj_pair("reg", writeReg, os);

  os << ",";

  auto writeSegments = [&LI, &os]() {
    for (auto &S : LI.segments) {
      write_segment(S, os);
    }
  };

  json_array("segments", writeSegments, os);

  os << ",";

  auto writeSubRanges = [&LI, &os]() {
    for (auto &SR : LI.subranges()) {
      write_subrange(SR, os);
    }
  };

  json_array("subranges", writeSubRanges, os);

  os << "},\n";
}

std::string get_legal_str(const MDString *MDStr) {
  std::string str;
  raw_string_ostream Stream(str);
  MDStr->print(Stream);
  Stream.flush();
  // Remove !.
  str = str.substr(1);
  // Remove ""
  str = str.substr(1);
  str.pop_back();
  std::replace(str.begin(), str.end(), '\\', '#');
  return str;
}

void write_file(const MDNode *FileNode, raw_ostream &os) {
  const MDString *FileName = cast<MDString>(FileNode->getOperand(0).get());
  StringRef fileNameStr = FileName->getString();
  if (fileNameStr.find("__AMDGPU_GPUMAP_") == 0)
    return;
  if (fileNameStr.find("__AMDGPU_DWARF_") == 0)
    return;

  os << "{";

  std::string str0 = get_legal_str(FileName);
  auto writeName = [&str0, &os]() { os << str0; };
  json_pair("filename", writeName, os);

  os << ",\n";

  const MDString *Content = cast<MDString>(FileNode->getOperand(1).get());
  std::string str = get_legal_str(Content);
  auto writeContent = [&str, &os]() { os << str; };
  json_pair("content", writeContent, os);
  os << "\n},\n";
}

void write_DIFile(const DIFile *File, raw_ostream &os) {
  if (File) {
    std::string name = get_legal_str(File->getRawFilename());
    std::string dir = "";
    if (MDString *MDDir = File->getRawDirectory())
      dir = get_legal_str(MDDir);
    os << dir << name;
  } else {
    os << "ArtificialFile";
  }
}

void write_line_mapping(SlotIndex Slot, DebugLoc DL, raw_ostream &os) {
  os << "{";

  auto writeSlot = [&Slot, &os]() { Slot.print(os); };

  json_pair("slot_index", writeSlot, os);

  os << ",\n";

  MDNode *Scope = DL.getScope();
  unsigned line = DL.getLine();
  unsigned col = DL.getCol();

  auto writeLine = [&line, &os]() { os << line; };
  json_pair("line", writeLine, os);

  os << ",\n";

  auto writeCol = [&col, &os]() { os << col; };
  json_pair("col", writeCol, os);

  os << ",\n";

  auto writeFile = [&Scope, &os]() {
    const DIFile *File = cast<DIScope>(Scope)->getFile();
    write_DIFile(File, os);
  };
  json_pair("file", writeFile, os);

  if (DILocation *inlineDL = DL.getInlinedAt()) {
    os << ",\n";
    unsigned inlineLine = inlineDL->getLine();
    auto writeLine = [&inlineLine, &os]() { os << inlineLine; };
    json_pair("inline_line", writeLine, os);

    os << ",\n";

    unsigned inlineCol = inlineDL->getColumn();
    auto writeCol = [&inlineCol, &os]() { os << inlineCol; };
    json_pair("inline_col", writeCol, os);

    os << ",\n";

    const MDNode *InlineScope = DL.getInlinedAtScope();
    auto writeFile = [&InlineScope, &os]() {
      const DIFile *File = cast<DIScope>(InlineScope)->getFile();
      write_DIFile(File, os);
    };
    json_pair("inline_file", writeFile, os);
  }

  os << "\n},\n";
}

void write_dbg_val(unsigned Reg, const DIVariable *V, const DIExpression *Exp,
                   const MachineRegisterInfo &MRI, const SIRegisterInfo *SIRI,
                   raw_ostream &os) {
  os << "{";

  auto writeReg = [&MRI, &SIRI, &Reg, &os]() {
    const unsigned SubReg = 0;
    write_reg(Reg, SubReg, MRI, SIRI, os);
  };
  json_obj_pair("reg", writeReg, os);

  os << ",\n";

  if (V) {
    auto writeName = [&V, &os]() { os << V->getName(); };
    json_pair("debug_val_name", writeName, os);
    os << ",\n";

    auto writeFile = [&V, &os]() {
      const DIFile *File = V->getFile();
      write_DIFile(File, os);
    };
    json_pair("debug_val_file", writeFile, os);
    os << ",\n";

    auto writeLine = [&V, &os]() { os << V->getLine(); };
    json_pair("debug_val_line", writeLine, os);
  }

  if (Exp->isValid() && Exp->getNumElements()) {
    os << ",\n";
    auto writeV = [&Exp, &os]() {
      os << '[';
      bool NeedSep = false;
      for (auto Op : Exp->expr_ops()) {
        if (NeedSep)
          os << ", ";
        else
          NeedSep = true;
        os << dwarf::OperationEncodingString(Op.getOp());
        for (unsigned I = 0; I < Op.getNumArgs(); ++I)
          os << ' ' << Op.getArg(I);
      }
      os << "] ";
    };
    json_pair("debug_exp", writeV, os);
  }
  os << "\n},\n";
}

void write_dbg_info(MachineFunction &MF, LiveIntervals *LIS,
                    const MachineRegisterInfo &MRI, const SIInstrInfo *SIII,
                    const SIRegisterInfo *SIRI, const SlotIndexes *SlotIndexes,
                    const NamedMDNode *SourceMD, raw_ostream &os) {
  os << ",\n";

  auto writeFiles = [&SourceMD, &os]() {
    for (const MDNode *FileNode : SourceMD->operands()) {
      write_file(FileNode, os);
    }
  };

  json_array("files", writeFiles, os);

  os << ",\n";

  auto writeLineMapping = [&MF, &SlotIndexes, &os]() {
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (MI.isDebugInstr()) {
          continue;
        }
        const DebugLoc DL = MI.getDebugLoc();
        if (!DL)
          continue;
        SlotIndex Slot = SlotIndexes->getInstructionIndex(MI);
        write_line_mapping(Slot, DL, os);
      }
    }
  };

  json_array("line_mapping", writeLineMapping, os);

  os << ",\n";

  auto writeDebugVals = [&MF, &MRI, &SIRI, &os]() {
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
        write_dbg_val(Reg.getReg(), V, Exp, MRI, SIRI, os);
      }
    }
  };

  json_array("debug_vals", writeDebugVals, os);
}

void write_function(MachineFunction &MF, LiveIntervals *LIS,
                    const MachineRegisterInfo &MRI, const SIInstrInfo *SIII,
                    const SIRegisterInfo *SIRI, raw_ostream &os) {
  const SlotIndexes *SlotIndexes = LIS->getSlotIndexes();

  os << "{\n";
  auto writeName = [&MF, &os]() { os << MF.getName(); };
  json_pair("name", writeName, os);

  os << ",\n";

  auto writeBlocks = [&MF, &SlotIndexes, &LIS, &MRI, &SIRI, &SIII, &os]() {
    for (MachineBasicBlock &MBB : MF) {
      write_block(MBB, LIS, MRI, SIRI, SIII, os);
      os << ",\n";
    }
  };

  json_array("blocks", writeBlocks, os);

  os << ",\n";

  auto writeDefines = [&MF, &SlotIndexes, &MRI, &SIRI, &os]() {
    write_defines(MF, SlotIndexes, MRI, SIRI, os);
  };

  json_array("defines", writeDefines, os);

  os << ",\n";

  auto writeUses = [&MF, &SlotIndexes, &MRI, &SIRI, &os]() {
    write_uses(MF, SlotIndexes, MRI, SIRI, os);
  };

  json_array("uses", writeUses, os);

  os << ",\n";

  auto writeLiveness = [&MF, &LIS, &MRI, &SIRI, &os]() {
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB) {
        if (MI.isDebugInstr())
          continue;
        const SlotIndex &SI = LIS->getInstructionIndex(MI).getBaseIndex();
        GCNRPTracker::LiveRegSet LISLR = llvm::getLiveRegs(SI, *LIS, MRI);
        write_liveness(SI, LISLR, MRI, SIRI, os);
      }
  };

  json_array("liveness", writeLiveness, os);

  os << ",\n";

  auto writeLiveIntervals = [&MRI, &SIRI, &LIS, &os]() {
    for (unsigned i = 0; i < MRI.getNumVirtRegs(); i++) {
      auto Reg = Register::index2VirtReg(i);
      if (!LIS->hasInterval(Reg))
        continue;
      auto &LI = LIS->getInterval(Reg);
      write_live_interval(LI, MRI, SIRI, os);
    }
  };

  json_array("live_intervals", writeLiveIntervals, os);

#if 0 // TODO: Do we need this?
  // Check debug info.
  const Function &F = MF.getFunction();
  const Module *M = F.getParent();
  const NamedMDNode *SourceMD =
      M->getNamedMetadata(hlsl::DxilMDHelper::kDxilSourceContentsMDName);
  if (SourceMD) {
    write_dbg_info(MF, LIS, MRI, SIII, SIRI, SlotIndexes, SourceMD, os);
  }
#endif

  os << "\n}";
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

void write_pressure(MachineFunction &MF, LiveIntervals *LIS, raw_ostream &os) {
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const auto *SIII = ST->getInstrInfo();
  const auto *SIRI = ST->getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  write_function(MF, LIS, MRI, SIII, SIRI, os);
  os.flush();
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
  void writeInst(MachineInstr &MI, const SIInstrInfo *SIII, raw_ostream &os);
  void writeBlock(MachineBasicBlock &MBB, const SIInstrInfo *SIII,
                  raw_ostream &os);
  void write(raw_ostream &os);
};

void buildMIContribution(MachineInstr &MI,
                         DenseSet<MachineInstr *> &ContributorSet,
                         DenseSet<MachineInstr *> &ContributedSet,
                         const SIRegisterInfo &SIRI, MachineRegisterInfo &MRI) {
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
  bool bUpdated = false;
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (auto *MBB : RPOT) {
    for (auto &MI : *MBB) {
      auto &contributors = MIContributorMap[&MI];
      unsigned size = contributors.size();
      DenseSet<MachineInstr *> parentContributors;
      for (auto *CMI : contributors) {
        auto &pContributors = MIContributorMap[CMI];
        parentContributors.insert(pContributors.begin(), pContributors.end());
      }
      contributors.insert(parentContributors.begin(), parentContributors.end());
      bUpdated |= size < contributors.size();
    }
  }
  return bUpdated;
}

void ContributionList::build() {
  // Build contribution.
  auto &MRI = MF.getRegInfo();
  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const auto *SIRI = ST->getRegisterInfo();
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      auto &contributors = MIContributorMap[&MI];
      auto &contributed = MIContributedToMap[&MI];
      buildMIContribution(MI, contributors, contributed, *SIRI, MRI);
    }
  }
  // propagate contribution.
  bool bUpdated = true;
  while (bUpdated) {
    bUpdated = propagateContribution();
  }
}

void ContributionList::writeInst(MachineInstr &MI, const SIInstrInfo *SIII,
                                 raw_ostream &os) {
  os << "\n{\n";
  unsigned ID = MIIndexMap[&MI];
  auto writeSlot = [&ID, &os]() { os << ID; };

  json_pair("ID", writeSlot, os);

  os << ",";

  auto writeAsm = [&MI, &SIII, &os]() {
    MI.print(os, /*IsStandalone*/ true, /*SkipOpers*/ false,
             /*SkipDebugLoc*/ true, /*AddNewLine*/ false, SIII);
  };
  json_pair("asm", writeAsm, os);

  os << ",\n";

  auto &contributors = MIContributorMap[&MI];
  auto writeContributor = [&contributors, this, &os]() {
    for (auto *MI : contributors) {
      unsigned ID = MIIndexMap[MI];
      os << ID << ",";
    }
  };

  json_array("contributors", writeContributor, os);
  os << ",\n";

  auto &contributeds = MIContributedToMap[&MI];
  auto writeContributed = [&contributeds, this, &os]() {
    for (auto *MI : contributeds) {
      unsigned ID = MIIndexMap[MI];
      os << ID << ",";
    }
  };

  json_array("contributed", writeContributed, os);
  os << "\n}\n";
}

void ContributionList::writeBlock(MachineBasicBlock &MBB,
                                  const SIInstrInfo *SIII, raw_ostream &os) {
  os << "{\n";
  auto writeName = [&MBB, &os]() { os << MBB.getName(); };
  json_pair("name", writeName, os);

  os << ",";

  auto writeIndex = [&MBB, &os]() { os << MBB.getNumber(); };
  json_pair("id", writeIndex, os);

  os << ",\n";

  auto writeInsts = [this, &MBB, &SIII, &os]() {
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      writeInst(MI, SIII, os);
      os << ",\n";
    }
  };

  json_array("instructions", writeInsts, os);

  os << ",\n";

  auto writePreds = [&MBB, &os]() {
    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      os << Pred->getNumber() << ",";
    }
  };

  json_array("preds", writePreds, os);

  os << ",";

  auto writeSuccs = [&MBB, &os]() {
    for (MachineBasicBlock *Succ : MBB.successors()) {
      os << Succ->getNumber() << ",";
    }
  };

  json_array("succs", writeSuccs, os);

  os << "}";
}

void ContributionList::write(raw_ostream &os) {
  unsigned ID = 0;
  // Build ID for write.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (auto *MBB : RPOT) {
    for (auto &MI : *MBB) {
      MIIndexMap[&MI] = ID++;
    }
  }

  const GCNSubtarget *ST = &MF.getSubtarget<GCNSubtarget>();
  const auto *SIII = ST->getInstrInfo();

  os << "{\n";
  auto writeName = [this, &os]() { os << MF.getName(); };
  json_pair("name", writeName, os);

  os << ",\n";

  auto writeBlocks = [this, &SIII, &RPOT, &os]() {
    for (auto *MBB : RPOT) {
      writeBlock(*MBB, SIII, os);
      os << ",\n";
    }
  };

  json_array("blocks", writeBlocks, os);

  os << "\n}";
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

static bool IsPhysReg(const MachineOperand &Op) {
  return Op.isReg() && Op.getReg().isPhysical();
}

// Sometimes split bb uses physical registers defined in BB, have to add them to
// live-in or the ir is malformed.
void llvm::UpdatePhysRegLiveInForBlock(MachineBasicBlock *NewBB,
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
      if (!IsPhysReg(Use) || !Use.isUse())
        continue;

      // Reserved regs do not need to be tracked through live-in sets.
      unsigned Reg = Use.getReg();
      if (Use.isImplicit() && MRI && MRI->isReserved(Reg))
        continue;

      if (!DefSet.count(Reg))
        NewBB->addLiveIn(Reg);
    }

    // Add all physical register defs (exlicit+implicit) to the def register
    // set.
    for (MachineOperand &Def : MI.operands()) {
      // Only process physreg defs.
      if (!IsPhysReg(Def) || !Def.isDef())
        continue;
      DefSet.insert(Def.getReg());
    }
  }
}

void llvm::BuildPhysRegLiveInForBlock(MachineBasicBlock *NewBB,
                                      SmallDenseSet<unsigned, 8> &LiveOutSet,
                                      const MachineRegisterInfo *MRI) {
  for (auto rit = NewBB->rbegin(); rit != NewBB->rend(); rit++) {
    auto &MI = *rit;
    // Add all physical register defs (exlicit+implicit) to the def register
    // set.
    for (MachineOperand &Def : MI.operands()) {
      // Only process physreg defs.
      if (!IsPhysReg(Def) || !Def.isDef())
        continue;
      LiveOutSet.erase(Def.getReg());
    }
    // Add all undefined physical registers to the live in set.
    for (MachineOperand &Use : MI.operands()) {
      // Only process physreg uses.
      if (!IsPhysReg(Use) || !Use.isUse())
        continue;

      // Reserved regs do not need to be tracked through live-in sets.
      unsigned Reg = Use.getReg();
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

MachineReg llvm::CreateVirtualRegForOperand(MachineOpcode Opcode,
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

MachineReg llvm::CreateVirtualDstReg(MachineOpcode Opcode,
                                     MachineFunction &MF) {
  return llvm::CreateVirtualRegForOperand(Opcode, 0, MF);
}

// Return true if the MI is a copy of exec.
// If true then sets pDst to the destination register.
bool llvm::IsExecCopy(const MachineInstr &MI, MachineReg Exec,
                      MachineReg *pDst) {
  enum { DST = 0, SRC = 1 };
  bool FoundCopy = false;
  if (MI.getOpcode() == AMDGPU::COPY || MI.getOpcode() == AMDGPU::S_MOV_B32 ||
      MI.getOpcode() == AMDGPU::S_MOV_B64) {
    const MachineOperand &Src = MI.getOperand(SRC);
    if (Src.isReg() && Src.getReg() == Exec) {
      FoundCopy = true;
    }
  }
#if 0 // TODO: Delete this.
    else if (MI.getOpcode() == AMDGPU::AMDGPU_GET_ENTRY_ACTIVE_MASK_PSEUDO ||
             MI.getOpcode() == AMDGPU::AMDGPU_GET_ENTRY_ACTIVE_MASK_PSEUDO_32)
    {
        FoundCopy = true;
    }
#endif

  if (FoundCopy) {
    *pDst = MI.getOperand(DST).getReg();
  }

  return FoundCopy;
}

llvm::MachineRegWithSubReg llvm::GetWqmEntryActiveMask(MachineFunction &MF) {
  llvm::MachineRegWithSubReg LiveLaneMask = {AMDGPU::NoRegister,
                                             AMDGPU::NoSubRegister};
  if (MachineInstr *MI = GetWqmEntryActiveMaskInst(MF)) {
    LiveLaneMask.Reg = MI->getOperand(0).getReg();
    LiveLaneMask.SubReg = MI->getOperand(0).getSubReg();
  }

  return LiveLaneMask;
}

MachineInstr *llvm::GetWqmEntryActiveMaskInst(MachineFunction &MF) {
#if 0 // TODO: Get rid of this
    // Look forward in the entry block for the SET_LIVE_LANE_MASK instruction.
    // This instruction is added by the SIWholeQuadMode pass.
    MachineBasicBlock &MBB = MF.front();
    for (MachineInstr &MI : MBB)
    {
        if (MI.getOpcode() == AMDGPU::AMDGPU_SET_LIVE_LANE_MASK ||
            MI.getOpcode() == AMDGPU::AMDGPU_SET_LIVE_LANE_MASK_32)
        {
            return &MI;
        }
    }
#endif

  return nullptr;
}

bool llvm::IsFetchShaderCall(const MachineInstr *MI) {
#if 0 // TODO: Get rid of this.
    return 
        MI->getOpcode() == AMDGPU::AMDGPU_CALL_FETCH_SHADER ||
        MI->getAMDGPUFlag(MachineInstr::AMDGPUMIFlag::FetchShaderCall);
#else
  return false;
#endif
}

bool llvm::IsSccLiveAt(llvm::MachineBasicBlock *MBB,
                       llvm::MachineBasicBlock::iterator MI) {
  const TargetRegisterInfo *TRI =
      MBB->getParent()->getRegInfo().getTargetRegisterInfo();
  for (auto it = MI; it != MBB->end(); ++it) {
    const MachineInstr &CurMI = *it;
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
MachineBasicBlock::iterator llvm::FindOrCreateInsertionPointForSccDef(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MI,
    const TargetRegisterInfo *TRI, const SIInstrInfo *TII,
    MachineRegisterInfo *MRI, SccDefInsertPointConstraintFlags Constraints) {
  // If SCC is dead at MI when we can use MI as the insert point.
  if (!llvm::IsSccLiveAt(MBB, MI)) {
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
    // it will also catch writes to the subregisters (e.g. exec_lo).
    if (CheckForExecWrite && It->modifiesRegister(AMDGPU::EXEC, TRI)) {
      break;
    }

    if (It->modifiesRegister(AMDGPU::SCC, TRI) &&
        !It->readsRegister(AMDGPU::SCC, TRI)) {
      return It->getIterator();
    }
  }

  // If no safe location can be found in the block we can save and restore
  // SCC around MI. There is no way to directly read or write SCC so we use
  // s_cselect to read the current value of SCC and s_cmp to write the saved
  // value back to SCC.
  //
  // The generated code will look like this;
  //
  //      S_CSELECT_B32 %SavedSCC, -1, 0  # Save SCC
  //      <----- Newly created safe insert point.
  //      MI
  //      S_CMP_LG_U32 %SavedSCC, 0       # Restore SCC
  //
  unsigned int TmpScc =
      MRI->createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);
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
                    SmallDenseSet<MachineBasicBlock *, 2> &touchedMBBSet) {
  MachineInstr *startMI = Indexes->getInstructionFromIndex(Seg->start);
  MachineInstr *endMI = Indexes->getInstructionFromIndex(Seg->end);
  // Treat non inst as not local.
  if (!startMI || !endMI)
    return false;
  // is local when parent MBB the same.
  bool bSameMBB = startMI->getParent() == endMI->getParent();
  if (!bSameMBB)
    return false;
  // Collect touched MBB.
  MachineBasicBlock *MBB = startMI->getParent();
  touchedMBBSet.insert(MBB);
  return true;
}

bool isLocalLiveRange(const LiveRange *Range, SlotIndexes *Indexes,
                      SmallDenseSet<MachineBasicBlock *, 2> &touchedMBBSet) {
  for (const LiveRange::Segment &Seg : Range->segments) {
    if (!isLocalSegment(&Seg, Indexes, touchedMBBSet))
      return false;
  }
  return true;
}

bool isLocalSegment(const LiveRange::Segment *Seg, SlotIndexes *Indexes) {
  MachineInstr *startMI = Indexes->getInstructionFromIndex(Seg->start);
  MachineInstr *endMI = Indexes->getInstructionFromIndex(Seg->end);
  // Treat non inst as not local.
  if (!startMI || !endMI)
    return false;
  // is local when parent MBB the same.
  return startMI->getParent() == endMI->getParent();
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
// touchedMBBSet is used for scheduling where local live interval could cross
// multiple regions, need to calculate livereg for each region inside touched
// MBB.
bool llvm::isLocalLiveInterval(
    const LiveInterval &LI, SlotIndexes *Indexes,
    SmallDenseSet<MachineBasicBlock *, 2> &touchedMBBSet) {
  if (LI.hasSubRanges()) {
    for (const auto &S : LI.subranges()) {
      if (!isLocalLiveRange(&S, Indexes, touchedMBBSet))
        return false;
    }
  }
  return isLocalLiveRange(&LI, Indexes, touchedMBBSet);
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
    if (llvm::GetNonDebugMBBEnd(BBEnd, MBB)) {
      auto SI = SlotIndexes->getInstructionIndex(*BBEnd);
      MBBOutputSlotMap[&MBB] = After ? SI.getDeadSlot() : SI.getBaseIndex();
    }
  }

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;

    LaneBitmask LiveMask;
    const auto &LI = LIS->getInterval(Reg);

    // Skip local live interval to make live input/ouput faster.
    if (llvm::isLocalLiveInterval(LI, SlotIndexes))
      continue;

    for (auto outputIt : MBBOutputSlotMap) {
      MachineBasicBlock *MBB = outputIt.first;
      auto SI = outputIt.second;

      auto LiveMask = getLiveLaneMask(Reg, SI, *LIS, MRI);
      if (LiveMask.any())
        MBBLiveMap[MBB][Reg] = LiveMask;
    }
  }
}

unsigned llvm::GetCurrentVGPRCount(llvm::MachineFunction &MF,
                                   const SIRegisterInfo *SIRI) {
  auto &MRI = MF.getRegInfo();
  for (MCPhysReg Reg : reverse(AMDGPU::VGPR_32RegClass.getRegisters())) {
    if (MRI.isPhysRegUsed(Reg)) {
      return SIRI->getHWRegIndex(Reg) - SIRI->getHWRegIndex(AMDGPU::VGPR0) + 1;
    }
  }
  return 0;
}

unsigned llvm::GetCurrentSGPRCount(llvm::MachineFunction &MF,
                                   const SIRegisterInfo *SIRI) {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned ScratchRSrcReg = MFI->getScratchRSrcReg();
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
  for (auto it : LiveSet) {
    int Reg = it.first;
    dbgs() << printReg(Reg, SIRI);
    if (it.second.any()) {
      dbgs() << " mask:" << it.second.getAsInteger();
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
llvm::FindSuccessor(llvm::MachineBasicBlock *MBB,
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
