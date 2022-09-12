//===- SIInsertWaterfall.cpp - insert waterall loops at intrinsic markers -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Replace 3 intrinsics used to mark waterfall regions with actual waterfall
/// loops. This is done at MachineIR level rather than LLVM-IR due to the use of
/// exec mask in this operation.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#include <set>

using namespace llvm;

#define DEBUG_TYPE "si-insert-waterfall"

namespace {

static unsigned getWFBeginSize(const unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::SI_WATERFALL_BEGIN_V1:
    return 1;
  case AMDGPU::SI_WATERFALL_BEGIN_V2:
    return 2;
  case AMDGPU::SI_WATERFALL_BEGIN_V4:
    return 4;
  case AMDGPU::SI_WATERFALL_BEGIN_V8:
    return 8;
  default:
    break;
  }

  return 0; // Not SI_WATERFALL_BEGIN_*
}

static unsigned getWFRFLSize(const unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::SI_WATERFALL_READFIRSTLANE_V1:
    return 1;
  case AMDGPU::SI_WATERFALL_READFIRSTLANE_V2:
    return 2;
  case AMDGPU::SI_WATERFALL_READFIRSTLANE_V4:
    return 4;
  case AMDGPU::SI_WATERFALL_READFIRSTLANE_V8:
    return 8;
  default:
    break;
  }

  return 0; // Not SI_WATERFALL_READFIRSTLANE_*
}

static unsigned getWFEndSize(const unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::SI_WATERFALL_END_V1:
    return 1;
  case AMDGPU::SI_WATERFALL_END_V2:
    return 2;
  case AMDGPU::SI_WATERFALL_END_V4:
    return 4;
  case AMDGPU::SI_WATERFALL_END_V8:
    return 8;
  default:
    break;
  }

  return 0; // Not SI_WATERFALL_END_*
}

static unsigned getWFLastUseSize(const unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::SI_WATERFALL_LAST_USE_V1:
    return 1;
  case AMDGPU::SI_WATERFALL_LAST_USE_V2:
    return 2;
  case AMDGPU::SI_WATERFALL_LAST_USE_V4:
    return 4;
  case AMDGPU::SI_WATERFALL_LAST_USE_V8:
    return 8;
  default:
    break;
  }

  return 0; // Not SI_WATERFALL_LAST_USE_*
}

static void readFirstLaneReg(MachineBasicBlock &MBB, MachineRegisterInfo *MRI,
                             const SIRegisterInfo *RI, const SIInstrInfo *TII,
                             MachineBasicBlock::iterator &I, const DebugLoc &DL,
                             Register RFLReg, Register RFLSrcReg,
                             const MachineOperand &RFLSrcOp) {
  auto RFLRegRC = MRI->getRegClass(RFLReg);
  uint32_t RegSize = RI->getRegSizeInBits(*RFLRegRC) / 32;
  assert(RI->hasVGPRs(MRI->getRegClass(RFLSrcReg)) && "unexpected uniform operand for readfirstlane");

  if (RegSize == 1)
    BuildMI(MBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), RFLReg)
        .addReg(RFLSrcReg, getUndefRegState(RFLSrcOp.isUndef()),
                RFLSrcOp.getSubReg());
  else {
    SmallVector<Register, 8> TRegs;
    for (unsigned i = 0; i < RegSize; ++i) {
      Register TReg = MRI->createVirtualRegister(&AMDGPU::SGPR_32RegClass);
      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), TReg)
          .addReg(RFLSrcReg, 0, RI->getSubRegFromChannel(i));
      TRegs.push_back(TReg);
    }
    MachineInstrBuilder MIB =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::REG_SEQUENCE), RFLReg);
    for (unsigned i = 0; i < RegSize; ++i) {
      MIB.addReg(TRegs[i]);
      MIB.addImm(RI->getSubRegFromChannel(i));
    }
  }
}

// Check if operand is uniform by checking:
// 1. Trivially detectable as operand in SGPR
// 2. Direct def is from an SGPR->VGPR copy (which may happen if assumed non-uniform value
//    turns out to be uniform)
static Register getUniformOperandReplacementReg(MachineRegisterInfo *MRI,
                                                const SIRegisterInfo *RI,
                                                Register Reg) {
  auto RegRC = MRI->getRegClass(Reg);
  if (!RI->hasVGPRs(RegRC)) {
    return Reg;
  }

  // Check for operand def being a copy from SGPR
  MachineInstr *DefMI = MRI->getVRegDef(Reg);
  if (DefMI->isFullCopy()) {
    auto const &DefSrcOp = DefMI->getOperand(1);
    if (DefSrcOp.isReg() && DefSrcOp.getReg().isVirtual()) {
      Register ReplaceReg = DefSrcOp.getReg();
      if (!RI->hasVGPRs(MRI->getRegClass(ReplaceReg)))
        return ReplaceReg;
    }
  }
  return AMDGPU::NoRegister;
}

static Register compareIdx(MachineBasicBlock &MBB, MachineRegisterInfo *MRI,
                           const SIRegisterInfo *RI, const SIInstrInfo *TII,
                           MachineBasicBlock::iterator &I, const DebugLoc &DL,
                           Register CurrentIdxReg,
                           const MachineOperand &IndexOp, Register CondReg,
                           bool IsWave32) {
  // Iterate over the index in dword chunks and'ing the result with the
  // CondReg
  // Optionally CondReg is passed in from a previous compareIdx call
  Register IndexReg = IndexOp.getReg();
  auto IndexRC = RI->getRegClassForOperandReg(*MRI, IndexOp);
  unsigned AndOpc =
      IsWave32 ? AMDGPU::S_AND_B32 : AMDGPU::S_AND_B64;
  const auto *BoolXExecRC = RI->getRegClass(AMDGPU::SReg_1_XEXECRegClassID);

  uint32_t RegSize = RI->getRegSizeInBits(*IndexRC) / 32;

  if (RegSize == 1) {
    Register TReg = MRI->createVirtualRegister(BoolXExecRC);
    BuildMI(MBB, I, DL, TII->get(AMDGPU::V_CMP_EQ_U32_e64), TReg)
        .addReg(CurrentIdxReg)
        .addReg(IndexReg, 0, IndexOp.getSubReg());

    if (CondReg != AMDGPU::NoRegister) {
      Register TReg2 = MRI->createVirtualRegister(BoolXExecRC);
      BuildMI(MBB, I, DL, TII->get(AndOpc), TReg2).addReg(CondReg).addReg(TReg);
      CondReg = TReg2;
    } else {
      CondReg = TReg;
    }
  } else {
    unsigned StartCount;
    Register TReg;
    if (CondReg != AMDGPU::NoRegister) {
      TReg = CondReg;
      StartCount = 0;
    } else {
      TReg = MRI->createVirtualRegister(BoolXExecRC);
      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_CMP_EQ_U32_e64), TReg)
          .addReg(CurrentIdxReg, 0, AMDGPU::sub0)
          .addReg(IndexReg, 0, AMDGPU::sub0);
      StartCount = 1;
    }

    for (unsigned i = StartCount; i < RegSize; ++i) {
      Register TReg2 = MRI->createVirtualRegister(BoolXExecRC);
      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_CMP_EQ_U32_e64), TReg2)
          .addReg(CurrentIdxReg, 0, RI->getSubRegFromChannel(i))
          .addReg(IndexReg, 0, RI->getSubRegFromChannel(i));
      Register TReg3 = MRI->createVirtualRegister(BoolXExecRC);
      BuildMI(MBB, I, DL, TII->get(AndOpc), TReg3)
          .addReg(TReg)
          .addReg(TReg2);
      TReg = TReg3;
    }
    CondReg = TReg;
  }
  return CondReg;
}

class SIInsertWaterfall : public MachineFunctionPass {
private:
  struct WaterfallWorkitem {
    const SIInstrInfo *TII;
    const MachineRegisterInfo *MRI;
    Register TokReg; // This is always the token from the last begin intrinsic
    MachineInstr *Final;

    std::vector<MachineInstr *> BeginList;
    std::vector<MachineInstr *> RFLList;
    std::vector<MachineInstr *> EndList;
    std::vector<MachineInstr *> LastUseList;

    // List of corresponding init, newdst and phi registers used in loop for
    // end pseudos
    std::vector<std::pair<Register, Register>> EndRegs;
    std::vector<Register> RFLRegs;

    WaterfallWorkitem() = default;
    WaterfallWorkitem(MachineInstr *_Begin, const SIInstrInfo *_TII,
                      MachineRegisterInfo *_MRI)
        : TII(_TII), MRI(_MRI), Final(nullptr) {

      auto TokMO = TII->getNamedOperand(*_Begin, AMDGPU::OpName::tok_ret);

      assert(tokIsStart(TII->getNamedOperand(*_Begin, AMDGPU::OpName::tok)) &&
             "first begin does not have an undefined input token as expected");
      assert(TokMO &&
             "Unable to extract tok operand from SI_WATERFALL_BEGIN pseudo op");

      BeginList.push_back(_Begin);
      TokReg = TokMO->getReg();
    }

    WaterfallWorkitem(const SIInstrInfo *_TII, MachineRegisterInfo *_MRI)
        : TII(_TII), MRI(_MRI), TokReg(AMDGPU::NoRegister), Final(nullptr) {}

    void processCandidate(MachineInstr *Cand) {
      unsigned Opcode = Cand->getOpcode();
      // Trivially end any waterfall intrinsic instructions
      if (getWFBeginSize(Opcode) || getWFRFLSize(Opcode) ||
          getWFEndSize(Opcode) || getWFLastUseSize(Opcode)) {
        // TODO: A new waterfall clause shouldn't overlap with any uses
        // tagged by a last_use intrinsic
        return;
      }

      // Iterate over the LastUseList to determine if this instruction has a
      // later use of a tagged last_use
      for (auto Use : LastUseList) {
        auto UseMO = TII->getNamedOperand(*Use, AMDGPU::OpName::dst);
        Register UseReg = UseMO->getReg();

        if (Cand->findRegisterUseOperand(UseReg))
          Final = Cand;
      }
    }

    MachineInstr *getDefInstr(const MachineOperand *MO) const {
      if (MO->isReg() && MRI->hasOneDef(MO->getReg())) {
        return (*MRI->def_begin(MO->getReg())).getParent();
      }
      return nullptr;
    }

    bool tokIsStart(const MachineOperand *MO) const {
      MachineInstr *defInstr = getDefInstr(MO);
      if (defInstr && defInstr->getOpcode() == AMDGPU::S_MOV_B32) {
        auto CopySrcOp = TII->getNamedOperand(*defInstr, AMDGPU::OpName::src0);
        if (CopySrcOp && CopySrcOp->isImm()) {
          if (CopySrcOp->getImm() == 0)
            return true;
        }
      }
      return false;
    }

    bool addCandidate(MachineInstr *Cand) {
      unsigned Opcode = Cand->getOpcode();

      assert((getWFBeginSize(Opcode) || getWFRFLSize(Opcode) ||
              getWFEndSize(Opcode) || getWFLastUseSize(Opcode)) &&
             "expected a waterfall instruction in addCandidate");

      auto CandTokMO = TII->getNamedOperand(*Cand, AMDGPU::OpName::tok);
      // There are a couple of scenarios at this point:
      // 1. Standard - there's already been a begin that's been processed and
      //               set up the WaterfallWorkItem. In which case the token is
      //               valid and needs to be checked to ensure well- formed
      //               waterfall groups.
      // 2. Begins removed - begins were uniform - and all of them have been
      //               removed. Need to process the rest of the instructions
      //               in the group, and verify that they have a undefined
      //               token
      if (TokReg == AMDGPU::NoRegister) {
        // All begins have been removed - continue to process the rest of the
        // grouping ready for them to be removed in the next stage
        assert(!getWFBeginSize(Opcode) &&
               "unexpected begin instruction for addCandidate");
        assert(tokIsStart(CandTokMO) &&
               "waterfall group with no begin doesn't have undef tok input");

        TokReg = CandTokMO->getReg();
      }
      if (CandTokMO->getReg() == TokReg) {
        if (getWFBeginSize(Opcode)) {
          auto TokRetMO = TII->getNamedOperand(*Cand, AMDGPU::OpName::tok_ret);
          assert(TokRetMO && "Unable to extract tok_ret operand from "
                             "SI_WATERFALL_BEGIN pseudo op");
          BeginList.push_back(Cand);
          TokReg = TokRetMO->getReg();
          return true;
        } else if (getWFRFLSize(Opcode)) {
          RFLList.push_back(Cand);
          return true;
        } else if (getWFEndSize(Opcode)) {
          EndList.push_back(Cand);
          Final = Cand;
          return true;
        } else {
          LastUseList.push_back(Cand);
          return true;
        }
      }
      LLVM_DEBUG(dbgs() << "malformed waterfall instruction group");
      return false;
    }
  };

  std::vector<WaterfallWorkitem> Worklist;

  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  MachineRegisterInfo *MRI;
  const SIRegisterInfo *RI;

public:
  static char ID;

  SIInsertWaterfall() : MachineFunctionPass(ID) {
    initializeSIInsertWaterfallPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool removeRedundantWaterfall(WaterfallWorkitem &Item);
  bool processWaterfall(MachineBasicBlock &MBB);

  Register getToken(MachineInstr *MI);

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // End anonymous namespace.

INITIALIZE_PASS(SIInsertWaterfall, DEBUG_TYPE, "SI Insert waterfalls", false,
                false)

char SIInsertWaterfall::ID = 0;

char &llvm::SIInsertWaterfallID = SIInsertWaterfall::ID;

FunctionPass *llvm::createSIInsertWaterfallPass() {
  return new SIInsertWaterfall;
}

bool SIInsertWaterfall::removeRedundantWaterfall(WaterfallWorkitem &Item) {
  // In some cases, the waterfall is actually redundant
  // If all the readfirstlane intrinsics are actually for uniform values and
  // the token used in the begin/end isn't used in anything else the waterfall
  // can be removed.
  // Alternatively, prior passes may have removed the readfirstlane intrinsics
  // altogether, in this case the begin/end intrinsics are now redundant and can
  // also be removed.
  // The readfirstlane intrinsics are replaced with the uniform source value,
  // the loop is removed and the defs in the end intrinsics are just replaced
  // with the input operands
  // We can also have cases where the begins are all removed (all the indices
  // were actually uniform).

  // First step is to identify any readfirstlane intrinsics that are actually
  // uniform - unless there are no begin instructions, in which case we always
  // remove
  bool LoopRemoved = false;
  unsigned Removed = 0;
  std::vector<MachineInstr *> NewRFLList;
  std::vector<MachineInstr *> ToRemoveRFLList;

  for (auto RFLMI : Item.RFLList) {
    auto RFLSrcOp = TII->getNamedOperand(*RFLMI, AMDGPU::OpName::src);
    auto RFLDstOp = TII->getNamedOperand(*RFLMI, AMDGPU::OpName::dst);
    Register RFLSrcReg = RFLSrcOp->getReg();
    Register RFLDstReg = RFLDstOp->getReg();

    Register ReplaceReg = getUniformOperandReplacementReg(MRI, RI, RFLSrcReg);
    if (ReplaceReg != AMDGPU::NoRegister) {
      MRI->replaceRegWith(RFLDstReg, ReplaceReg);
      Removed++;
      ToRemoveRFLList.push_back(RFLMI);
    } else if (RFLDstOp->isDead()) {
      Removed++;
      ToRemoveRFLList.push_back(RFLMI);
    } else {
      NewRFLList.push_back(RFLMI);
    }
  }

  // Note: this test also returns true when there are NO RFL intrinsics, the
  // case where a prior pass has removed all of them and the loop is now redundant
  if (!Item.BeginList.size() || Removed == Item.RFLList.size()) {
    // Removed all of the RFLs
    // We can remove the waterfall loop entirely

    // Protocol is to replace all dst operands for the waterfall_end intrinsics with
    // their src operands.
    // Replace all last_use dst operands with their src operands
    // We don't need to check that the loop index isn't used anywhere as the
    // protocol for waterfall intrinsics is to only use the begin index via a
    // readfirstlane intrinsic anyway (which should also be removed)
    // Any problems due to errors in this pass around loop removal will be
    // picked up later by e.g. use before def errors
    LLVM_DEBUG(dbgs() << "detected case for waterfall loop removal - already all uniform\n");
    for (auto EndMI : Item.EndList) {
      auto EndDstReg = TII->getNamedOperand(*EndMI, AMDGPU::OpName::dst)->getReg();
      auto EndSrcReg = TII->getNamedOperand(*EndMI, AMDGPU::OpName::src)->getReg();
      MRI->replaceRegWith(EndDstReg, EndSrcReg);
    }
    for (auto LUMI : Item.LastUseList) {
      auto LUDstReg = TII->getNamedOperand(*LUMI, AMDGPU::OpName::dst)->getReg();
      auto LUSrcReg = TII->getNamedOperand(*LUMI, AMDGPU::OpName::src)->getReg();
      MRI->replaceRegWith(LUDstReg, LUSrcReg);
    }
    // If all the begins were removed, we have to replace the RFL with actual
    // RFL, these will show up in the NewRLFList
    for (auto RFLMI : NewRFLList) {
      auto DstReg = TII->getNamedOperand(*RFLMI, AMDGPU::OpName::dst)->getReg();
      auto SrcOp = TII->getNamedOperand(*RFLMI, AMDGPU::OpName::src);
      Register SrcReg = SrcOp->getReg();

      MachineBasicBlock::iterator RFLInsert(RFLMI);
      readFirstLaneReg(*RFLMI->getParent(), MRI, RI, TII, RFLInsert,
                       RFLMI->getDebugLoc(), DstReg, SrcReg, *SrcOp);
    }

    for (auto BeginMI : Item.BeginList)
      BeginMI->eraseFromParent();
    for (auto RFLMI : Item.RFLList)
      RFLMI->eraseFromParent();
    for (auto ENDMI : Item.EndList)
      ENDMI->eraseFromParent();
    for (auto LUMI : Item.LastUseList)
      LUMI->eraseFromParent();

    LoopRemoved = true;

  } else if (Removed) {
    LLVM_DEBUG(dbgs() << "Removed " << Removed <<
               " waterfall rfl intrinsics due to being uniform - updating remaining rfl list\n");
    // TODO: there's an opportunity to pull the DAG involving the (removed) rfls
    // out of the waterfall loop
    Item.RFLList.clear();
    std::copy(NewRFLList.begin(), NewRFLList.end(), std::back_inserter(Item.RFLList));
    for (auto RFLMI : ToRemoveRFLList)
      RFLMI->eraseFromParent();
  }

  return LoopRemoved;
}

bool SIInsertWaterfall::processWaterfall(MachineBasicBlock &MBB) {
  bool Changed = false;
  MachineFunction &MF = *MBB.getParent();
  MachineBasicBlock *CurrMBB = &MBB;

  // Firstly we check that there are at least 3 related waterfall instructions
  // for this begin
  // SI_WATERFALL_BEGIN [ SI_WATERFALL_BEGIN ]*
  // [ SI_WATERFALL_READFIRSTLANE ]+ [ SI_WATERFALL_END ]+ If there are multiple
  // waterfall loops they must also be disjoint

  for (WaterfallWorkitem &Item : Worklist) {
    LLVM_DEBUG(
        if (Item.BeginList.size()) {
          dbgs() << "Processing " << *Item.BeginList[0] << "\n";

          for (auto RUse = MRI->use_begin(Item.TokReg), RSE = MRI->use_end();
               RUse != RSE; ++RUse) {
            MachineInstr *RUseMI = RUse->getParent();
            assert((CurrMBB->getNumber() == RUseMI->getParent()->getNumber()) &&
                   "Linked WATERFALL pseudo ops found in different BBs");
          }
        } else { dbgs() << "Processing redundant waterfall\n"; });

    if (removeRedundantWaterfall(Item)) {
      Changed = true;
      continue;
    }

    assert(Item.RFLList.size() &&
           (Item.EndList.size() || Item.LastUseList.size()) &&
           "SI_WATERFALL* pseudo instruction group must have at least 1 of "
           "each type");

    // Insert the waterfall loop code around the identified region of
    // instructions
    // Loop starts at the last SI_WATERFALL_BEGIN
    // SI_WATERFALL_READFIRSTLANE is replaced with appropriate readfirstlane
    // instructions OR is removed
    // if the readfirstlane is using the same index as the SI_WATERFALL_BEGIN
    // Loop is ended after the last SI_WATERFALL_END and these instructions are
    // removed with the src replacing all dst uses
    typedef struct {
      const MachineOperand *Index;
      const TargetRegisterClass *IndexRC;
      const TargetRegisterClass *IndexSRC;
      Register CurrentIdxReg;
    } IdxInfo;

    std::vector<IdxInfo> IndexList;
#ifndef NDEBUG
    bool IsUniform = true;
#endif
    for (auto BeginMI : Item.BeginList) {
      IdxInfo CurrIdx;
      CurrIdx.Index = TII->getNamedOperand(*(BeginMI), AMDGPU::OpName::idx);
      CurrIdx.IndexRC = RI->getRegClassForOperandReg(*MRI, *CurrIdx.Index);
      CurrIdx.IndexSRC = RI->getEquivalentSGPRClass(CurrIdx.IndexRC);
      IndexList.push_back(CurrIdx);

      LLVM_DEBUG(if (RI->hasVGPRs(CurrIdx.IndexRC))
        IsUniform = false;);
    }

    LLVM_DEBUG(if (IsUniform) {
      // Waterfall loop index is uniform! Loop can be removed
      // TODO:: Implement loop removal
      dbgs() << "Uniform loop detected - waterfall loop is redundant\n";
    });

    MachineBasicBlock::iterator I(Item.BeginList.back());
    const DebugLoc &DL = Item.BeginList[0]->getDebugLoc();

    // Initialize the register we accumulate the result into, which is the
    // target of any SI_WATERFALL_END instruction
    for (auto EndMI : Item.EndList) {
      Register EndDst =
          TII->getNamedOperand(*EndMI, AMDGPU::OpName::dst)->getReg();
      Register EndSrc =
          TII->getNamedOperand(*EndMI, AMDGPU::OpName::src)->getReg();
      Item.EndRegs.emplace_back(EndDst, EndSrc);
    }
    for (auto LUMI : Item.LastUseList) {
      Register LUSrc =
          TII->getNamedOperand(*LUMI, AMDGPU::OpName::src)->getReg();
      Register LUDst =
          TII->getNamedOperand(*LUMI, AMDGPU::OpName::dst)->getReg();
      MRI->replaceRegWith(LUDst, LUSrc);
    }

    // EXEC mask handling
    Register Exec = ST->isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
    unsigned SaveExecOpc =
      ST->isWave32() ? AMDGPU::S_AND_SAVEEXEC_B32 : AMDGPU::S_AND_SAVEEXEC_B64;
    unsigned XorTermOpc =
      ST->isWave32() ? AMDGPU::S_XOR_B32_term : AMDGPU::S_XOR_B64_term;
    unsigned MovOpc =
      ST->isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
    const auto *BoolXExecRC = RI->getRegClass(AMDGPU::SReg_1_XEXECRegClassID);

    Register SaveExec = MRI->createVirtualRegister(BoolXExecRC);
    Register TmpExec = MRI->createVirtualRegister(BoolXExecRC);

    BuildMI(*CurrMBB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), TmpExec);

    // Save the EXEC mask
    BuildMI(*CurrMBB, I, DL, TII->get(MovOpc), SaveExec)
        .addReg(Exec);

    MachineBasicBlock &LoopBB = *MF.CreateMachineBasicBlock();
    MachineBasicBlock &RemainderBB = *MF.CreateMachineBasicBlock();
    MachineFunction::iterator MBBI(*CurrMBB);
    ++MBBI;

    MF.insert(MBBI, &LoopBB);
    MF.insert(MBBI, &RemainderBB);

    LoopBB.addSuccessor(&LoopBB);
    LoopBB.addSuccessor(&RemainderBB);

    // Move all instructions from the SI_WATERFALL_BEGIN to the last
    // SI_WATERFALL_END or last use tagged from SI_WATERFALL_LAST_USE
    // into the new LoopBB
    MachineBasicBlock::iterator SpliceE(Item.Final);
    ++SpliceE;
    LoopBB.splice(LoopBB.begin(), CurrMBB, I, SpliceE);

    // Iterate over the instructions inserted into the loop
    // Need to unset any kill flag on any uses as now this is a loop that is no
    // longer valid
    for (MachineInstr &MI : LoopBB)
      MI.clearKillInfo();

    RemainderBB.transferSuccessorsAndUpdatePHIs(CurrMBB);
    RemainderBB.splice(RemainderBB.begin(), CurrMBB, SpliceE, CurrMBB->end());
    MachineBasicBlock::iterator E(Item.Final);
    ++E;

    CurrMBB->addSuccessor(&LoopBB);

    MachineBasicBlock::iterator J = LoopBB.begin();

    Register PhiExec = MRI->createVirtualRegister(BoolXExecRC);
    Register NewExec = MRI->createVirtualRegister(BoolXExecRC);

    for (auto &CurrIdx : IndexList)
      CurrIdx.CurrentIdxReg = MRI->createVirtualRegister(CurrIdx.IndexSRC);

    BuildMI(LoopBB, J, DL, TII->get(TargetOpcode::PHI), PhiExec)
        .addReg(TmpExec)
        .addMBB(CurrMBB)
        .addReg(NewExec)
        .addMBB(&LoopBB);

    // Get the next index to use from the first enabled lane
    for (auto &CurrIdx : IndexList)
      readFirstLaneReg(LoopBB, MRI, RI, TII, J, DL, CurrIdx.CurrentIdxReg,
                       CurrIdx.Index->getReg(), *CurrIdx.Index);

    // Also process the readlane pseudo ops - if readfirstlane is using the
    // index then just replace with the CurrentIdxReg instead
    for (auto RFLMI : Item.RFLList) {
      auto RFLSrcOp = TII->getNamedOperand(*RFLMI, AMDGPU::OpName::src);
      auto RFLDstOp = TII->getNamedOperand(*RFLMI, AMDGPU::OpName::dst);
      Register RFLSrcReg = RFLSrcOp->getReg();
      Register RFLDstReg = RFLDstOp->getReg();

      bool MatchedIdx = false;
      for (auto &CurrIdx : IndexList) {
        if (RFLSrcReg == CurrIdx.Index->getReg()) {
          // Use the CurrentIdxReg for this
          Item.RFLRegs.push_back(CurrIdx.CurrentIdxReg);
          MRI->replaceRegWith(RFLDstReg, CurrIdx.CurrentIdxReg);
          MatchedIdx = true;
          break;
        }
      }
      if (!MatchedIdx) {
        Item.RFLRegs.push_back(RFLDstReg);
        // Insert function to expand to required size here
        MachineBasicBlock::iterator RFLInsert(RFLMI);
        readFirstLaneReg(LoopBB, MRI, RI, TII, RFLInsert, DL, RFLDstReg, RFLSrcReg,
                         *RFLSrcOp);
      }
    }

    // Compare the just read idx value to all possible idx values
    Register CondReg = AMDGPU::NoRegister;
    for (auto &CurrIdx : IndexList)
      CondReg = compareIdx(LoopBB, MRI, RI, TII, J, DL, CurrIdx.CurrentIdxReg,
                           *CurrIdx.Index, CondReg, ST->isWave32());

    // Update EXEC, save the original EXEC value to VCC
    BuildMI(LoopBB, J, DL, TII->get(SaveExecOpc), NewExec)
        .addReg(CondReg, RegState::Kill);

    // TODO: Conditional branch here to loop header as potential optimization?

    // Copy the just read value into the destination
    for (auto EndReg : Item.EndRegs) {
      MachineBasicBlock::iterator EndInsert(Item.Final);
      BuildMI(LoopBB, EndInsert, DL, TII->get(AMDGPU::COPY), EndReg.first)
          .addReg(EndReg.second);
    }

    MRI->setSimpleHint(NewExec, CondReg);

    // Update EXEC, switch all done bits to 0 and all todo bits to 1.
    BuildMI(LoopBB, E, DL, TII->get(XorTermOpc), Exec)
        .addReg(Exec)
        .addReg(NewExec);

    // XXX - s_xor_b64 sets scc to 1 if the result is nonzero, so can we use
    // s_cbranch_scc0?

    // Loop back if there are still variants to cover
    BuildMI(LoopBB, E, DL, TII->get(AMDGPU::SI_WATERFALL_LOOP)).addMBB(&LoopBB);

    MachineBasicBlock::iterator First = RemainderBB.begin();
    BuildMI(RemainderBB, First, DL, TII->get(MovOpc), Exec)
        .addReg(SaveExec);

    for (auto BeginMI : Item.BeginList)
      BeginMI->eraseFromParent();
    for (auto RFLMI : Item.RFLList)
      RFLMI->eraseFromParent();
    for (auto ENDMI : Item.EndList)
      ENDMI->eraseFromParent();
    for (auto LUMI : Item.LastUseList)
      LUMI->eraseFromParent();

    // To process subsequent waterfall groups, update CurrMBB to the RemainderBB
    CurrMBB = &RemainderBB;

    Changed = true;
  }
  return Changed;
}

Register SIInsertWaterfall::getToken(MachineInstr *MI) {
  auto CandTokMO = TII->getNamedOperand(*MI, AMDGPU::OpName::tok);
  return CandTokMO->isReg() ? CandTokMO->getReg() : AMDGPU::NoRegister;
}

bool SIInsertWaterfall::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  RI = ST->getRegisterInfo();

  for (MachineBasicBlock &MBB : MF) {
    Worklist.clear();
    bool StartNew = true;

    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();

      if (getWFBeginSize(Opcode)) {
        if (StartNew) {
          Worklist.push_back(WaterfallWorkitem(&MI, TII, MRI));
          StartNew = false;
        } else {
          if (!Worklist.back().addCandidate(&MI)) {
            llvm_unreachable("Incorrect SI_WATERFALL_* groups");
          }
        }
      } else if (getWFRFLSize(Opcode) || getWFEndSize(Opcode) ||
                 getWFLastUseSize(Opcode)) {
        // On to the body of the group intrinsics,

        // Tag StartNew as true if we encounter another begin
        StartNew = true;

        if (!Worklist.size() || getToken(&MI) != Worklist.back().TokReg) {
          // There's no associated begin for these body intrinsics
          // That means it's either an error - or all the begin intrinsics
          // were removed due to being uniform
          // Set up a WorkItem so we can process this correctly
          Worklist.push_back(WaterfallWorkitem(TII, MRI));
        }

        if (!Worklist.back().addCandidate(&MI)) {
          llvm_unreachable("Overlapping SI_WATERFALL_* groups");
        }
      } else {
        if (Worklist.size())
          Worklist.back().processCandidate(&MI);
      }
    }
    Changed |= processWaterfall(MBB);
  }

  return Changed;
}
