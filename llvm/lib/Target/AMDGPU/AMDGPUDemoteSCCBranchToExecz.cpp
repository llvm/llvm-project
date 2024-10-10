#include <llvm/CodeGen/MachineFunctionPass.h>

#include "AMDGPU.h"
#include "AMDGPUDemoteSCCBranchToExecz.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"

using namespace llvm;

namespace {
#define DEBUG_TYPE "amdgpu-demote-scc-to-execz"
const char PassName[] = "AMDGPU s_cbranch_scc to s_cbranch_execz conversion";

std::optional<unsigned> getVALUOpc(const MachineInstr &MI,
                                   bool Reverse = false) {
  unsigned Opc = MI.getOpcode();
  if (Reverse) {
    switch (Opc) {
    case AMDGPU::S_CMP_EQ_I32:
      Opc = AMDGPU::S_CMP_LG_I32;
      break;
    case AMDGPU::S_CMP_LG_I32:
      Opc = AMDGPU::S_CMP_EQ_I32;
      break;
    case AMDGPU::S_CMP_GT_I32:
      Opc = AMDGPU::S_CMP_LE_I32;
      break;
    case AMDGPU::S_CMP_GE_I32:
      Opc = AMDGPU::S_CMP_LT_I32;
      break;
    case AMDGPU::S_CMP_LT_I32:
      Opc = AMDGPU::S_CMP_GE_I32;
      break;
    case AMDGPU::S_CMP_LE_I32:
      Opc = AMDGPU::S_CMP_GT_I32;
      break;
    case AMDGPU::S_CMP_EQ_U32:
      Opc = AMDGPU::S_CMP_LG_U32;
      break;
    case AMDGPU::S_CMP_LG_U32:
      Opc = AMDGPU::S_CMP_EQ_U32;
      break;
    case AMDGPU::S_CMP_GT_U32:
      Opc = AMDGPU::S_CMP_LE_U32;
      break;
    case AMDGPU::S_CMP_GE_U32:
      Opc = AMDGPU::S_CMP_LT_U32;
      break;
    case AMDGPU::S_CMP_LT_U32:
      Opc = AMDGPU::S_CMP_GE_U32;
      break;
    case AMDGPU::S_CMP_LE_U32:
      Opc = AMDGPU::S_CMP_GT_U32;
      break;
    case AMDGPU::S_CMP_EQ_U64:
      Opc = AMDGPU::S_CMP_LG_U64;
      break;
    case AMDGPU::S_CMP_LG_U64:
      Opc = AMDGPU::S_CMP_EQ_U64;
      break;
    default:
      return std::nullopt;
    }
  }

  switch (Opc) {
  case AMDGPU::S_CMP_EQ_I32:
    return AMDGPU::V_CMP_EQ_I32_e64;
  case AMDGPU::S_CMP_LG_I32:
    return AMDGPU::V_CMP_LT_I32_e64;
  case AMDGPU::S_CMP_GT_I32:
    return AMDGPU::V_CMP_GT_I32_e64;
  case AMDGPU::S_CMP_GE_I32:
    return AMDGPU::V_CMP_GE_I32_e64;
  case AMDGPU::S_CMP_LT_I32:
    return AMDGPU::V_CMP_LT_I32_e64;
  case AMDGPU::S_CMP_LE_I32:
    return AMDGPU::V_CMP_LE_I32_e64;
  case AMDGPU::S_CMP_EQ_U32:
    return AMDGPU::V_CMP_EQ_U32_e64;
  case AMDGPU::S_CMP_LG_U32:
    return AMDGPU::V_CMP_NE_U32_e64;
  case AMDGPU::S_CMP_GT_U32:
    return AMDGPU::V_CMP_GT_U32_e64;
  case AMDGPU::S_CMP_GE_U32:
    return AMDGPU::V_CMP_GE_U32_e64;
  case AMDGPU::S_CMP_LT_U32:
    return AMDGPU::V_CMP_LT_U32_e64;
  case AMDGPU::S_CMP_LE_U32:
    return AMDGPU::V_CMP_LE_U32_e64;
  case AMDGPU::S_CMP_EQ_U64:
    return AMDGPU::V_CMP_EQ_U64_e64;
  case AMDGPU::S_CMP_LG_U64:
    return AMDGPU::V_CMP_NE_U64_e64;
  default:
    return std::nullopt;
  }
}

bool isSCmpPromotableToVCmp(const MachineInstr &MI) {
  return getVALUOpc(MI).has_value();
}

bool isTriangular(MachineBasicBlock &Head, MachineBasicBlock *&Then,
                  MachineBasicBlock *&Tail) {
  if (Head.succ_size() != 2)
    return false;

  Then = Head.succ_begin()[0];
  Tail = Head.succ_begin()[1];

  // Canonicalize so Succ0 has MBB as its single predecessor.
  if (Then->pred_size() != 1)
    std::swap(Then, Tail);

  if (Then->pred_size() != 1 || Then->succ_size() != 1)
    return false;

  return *Then->succ_begin() == Tail;
}

bool hasPromotableCmpConditon(MachineInstr &Term, MachineInstr *&Cmp) {
  auto CmpIt = std::next(Term.getReverseIterator());
  if (CmpIt == Term.getParent()->instr_rend())
    return false;

  if (!isSCmpPromotableToVCmp(*CmpIt))
    return false;

  Cmp = &*CmpIt;
  return true;
}

bool hasCbranchSCCTerm(MachineBasicBlock &Head, MachineInstr *&Term) {
  auto TermIt = Head.getFirstInstrTerminator();
  if (TermIt == Head.end())
    return false;

  switch (TermIt->getOpcode()) {
  case AMDGPU::S_CBRANCH_SCC0:
  case AMDGPU::S_CBRANCH_SCC1:
    Term = &*TermIt;
    return true;
  default:
    return false;
  }
}

bool isTriangularSCCBranch(MachineBasicBlock &Head, MachineInstr *&Term,
                           MachineInstr *&Cmp, MachineBasicBlock *&Then,
                           MachineBasicBlock *&Tail) {

  if (!hasCbranchSCCTerm(Head, Term))
    return false;

  if (!isTriangular(Head, Then, Tail))
    return false;

  // phi-nodes in the tail can prevent splicing the instructions of the then
  // and tail blocks in the head
  if (!Tail->empty() && Tail->begin()->isPHI())
    return false;

  if (!hasPromotableCmpConditon(*Term, Cmp))
    return false;

  return true;
}

bool SCC1JumpsToThen(const MachineInstr &Term, const MachineBasicBlock &Then) {
  MachineBasicBlock *TBB = Term.getOperand(0).getMBB();
  return (TBB == &Then) == (Term.getOpcode() == AMDGPU::S_CBRANCH_SCC1);
}

class AMDGPUDemoteSCCBranchToExecz {
  MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &RegInfo;
  const TargetSchedModel &SchedModel;

public:
  AMDGPUDemoteSCCBranchToExecz(MachineFunction &MF)
      : MF(MF), ST(MF.getSubtarget<GCNSubtarget>()), TII(*ST.getInstrInfo()),
        RegInfo(*ST.getRegisterInfo()), SchedModel(TII.getSchedModel()) {}

  bool mustRetainSCCBranch(const MachineInstr &Term, const MachineInstr &Cmp,
                           const MachineBasicBlock &Then,
                           const MachineBasicBlock &Tail) {
    bool IsWave32 = TII.isWave32();
    unsigned AndSaveExecOpc =
        IsWave32 ? AMDGPU::S_AND_SAVEEXEC_B32 : AMDGPU::S_AND_SAVEEXEC_B64;
    unsigned Mov = IsWave32 ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
    unsigned NewOps[] = {*getVALUOpc(Cmp, !SCC1JumpsToThen(Term, Then)),
                         AndSaveExecOpc, Mov};
    unsigned NewOpsCost = 0;
    for (unsigned Opc : NewOps)
      NewOpsCost += SchedModel.computeInstrLatency(Opc);
    unsigned OldCmpCost = SchedModel.computeInstrLatency(&Cmp, false);

    assert(NewOpsCost >= OldCmpCost);
    return !TII.mustRetainExeczBranch(Term, Then, Tail,
                                      NewOpsCost - OldCmpCost);
  }

  void demoteCmp(MachineInstr &Term, MachineInstr &Cmp, MachineBasicBlock &Head,
                 MachineBasicBlock &Then, MachineBasicBlock &Tail) {
    unsigned NewCmpOpc = *getVALUOpc(Cmp, !SCC1JumpsToThen(Term, Then));
    Cmp.setDesc(TII.get(NewCmpOpc));

    MachineOperand L = Cmp.getOperand(0);
    MachineOperand R = Cmp.getOperand(1);
    for (unsigned i = 3; i != 0; --i)
      Cmp.removeOperand(i - 1);

    auto VCC = RegInfo.getVCC();
    auto Exec = RegInfo.getExec();

    auto &MRI = MF.getRegInfo();
    MCRegister ExecBackup =
        MRI.createVirtualRegister(RegInfo.getPhysRegBaseClass(Exec));

    Cmp.addOperand(MachineOperand::CreateReg(VCC, true));
    Cmp.addOperand(L);
    Cmp.addOperand(R);
    Cmp.addImplicitDefUseOperands(MF);

    TII.legalizeOperands(Cmp);

    bool IsWave32 = TII.isWave32();
    unsigned AndSaveExecOpc =
        IsWave32 ? AMDGPU::S_AND_SAVEEXEC_B32 : AMDGPU::S_AND_SAVEEXEC_B64;
    auto SaveAndMaskExec = BuildMI(*Term.getParent(), Term, Cmp.getDebugLoc(),
                                   TII.get(AndSaveExecOpc), ExecBackup);
    SaveAndMaskExec.addReg(VCC, RegState::Kill);
    SaveAndMaskExec->getOperand(3).setIsDead(); // mark SCC as dead

    DebugLoc DL = Term.getDebugLoc();
    TII.removeBranch(Head);
    MachineOperand Cond[] = {
        MachineOperand::CreateImm(SIInstrInfo::BranchPredicate::EXECZ),
        MachineOperand::CreateReg(RegInfo.getExec(), false)};
    TII.insertBranch(Head, &Tail, &Then, Cond, DL);

    TII.restoreExec(MF, Tail, Tail.instr_begin(), DebugLoc(), ExecBackup);
  }

  bool run() {
    if (!SchedModel.hasInstrSchedModel())
      return false;
    bool Changed = false;

    for (MachineBasicBlock &Head : MF) {
      MachineInstr *Term;
      MachineInstr *Cmp;
      MachineBasicBlock *Then;
      MachineBasicBlock *Tail;
      if (!isTriangularSCCBranch(Head, Term, Cmp, Then, Tail))
        continue;

      if (!mustRetainSCCBranch(*Term, *Cmp, *Then, *Tail))
        continue;

      demoteCmp(*Term, *Cmp, Head, *Then, *Tail);
      Changed = true;
    }
    return Changed;
  }
};

class AMDGPUDemoteSCCBranchToExeczLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUDemoteSCCBranchToExeczLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    AMDGPUDemoteSCCBranchToExecz IfCvt{MF};
    return IfCvt.run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PassName; }
};

char AMDGPUDemoteSCCBranchToExeczLegacy::ID = 0;

} // namespace

PreservedAnalyses llvm::AMDGPUDemoteSCCBranchToExeczPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  AMDGPUDemoteSCCBranchToExecz IfCvt{MF};
  if (!IfCvt.run())
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

char &llvm::AMDGPUDemoteSCCBranchToExeczLegacyID =
    AMDGPUDemoteSCCBranchToExeczLegacy::ID;
INITIALIZE_PASS_BEGIN(AMDGPUDemoteSCCBranchToExeczLegacy, DEBUG_TYPE, PassName,
                      false, false)
INITIALIZE_PASS_END(AMDGPUDemoteSCCBranchToExeczLegacy, DEBUG_TYPE, PassName,
                    false, false)
