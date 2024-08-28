#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SSAIfConv.h"

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-if-cvt"

namespace {
unsigned getReversedVCMPXOpcode(unsigned Opcode) {
  // TODO: this is a placeholder for the real function
  switch (Opcode) {
  case AMDGPU::V_CMPX_LT_I32_nosdst_e64:
    return AMDGPU::V_CMPX_GE_I32_nosdst_e64;
  default:
    errs() << "unhandled: " << Opcode << "\n";
    llvm_unreachable("unhandled vcmp opcode");
  }
}

bool needsExecPredication(const SIInstrInfo *TII, const MachineInstr &I) {
  return TII->isVALU(I) || TII->isVMEM(I);
}

struct ExecPredicate : SSAIfConv::PredicationStrategyBase {
  const SIInstrInfo *TII;
  const SIRegisterInfo *RegInfo;

  MachineInstr *Cmp = nullptr;

  ExecPredicate(const SIInstrInfo *TII)
      : TII(TII), RegInfo(&TII->getRegisterInfo()) {}

  bool canConvertIf(MachineBasicBlock *Head, MachineBasicBlock *TBB,
                    MachineBasicBlock *FBB, MachineBasicBlock *Tail,
                    ArrayRef<MachineOperand> Cond) override {

    // check that the cmp is just before the branch and that it is promotable to
    // v_cmpx
    const unsigned SupportedBranchOpc[]{
        AMDGPU::S_CBRANCH_SCC0, AMDGPU::S_CBRANCH_SCC1, AMDGPU::S_CBRANCH_VCCNZ,
        AMDGPU::S_CBRANCH_VCCZ};

    MachineInstr &CBranch = *Head->getFirstInstrTerminator();
    if (!llvm::is_contained(SupportedBranchOpc, CBranch.getOpcode()))
      return false;

    auto CmpInstr = std::next(CBranch.getReverseIterator());
    if (CmpInstr == Head->instr_rend())
      return false;

    Register SCCorVCC = Cond[1].getReg();
    bool ModifiesConditionReg = CmpInstr->modifiesRegister(SCCorVCC, RegInfo);
    if (!ModifiesConditionReg)
      return false;

    Cmp = &*CmpInstr;

    unsigned CmpOpc = Cmp->getOpcode();
    if (TII->isSALU(*Cmp))
      CmpOpc = TII->getVALUOp(*Cmp);
    if (AMDGPU::getVCMPXOpFromVCMP(CmpOpc) == -1) {
      errs() << "unhandled branch " << *Cmp << "\n";
      return false;
    }

    return true;
  }

  bool canPredicateInstr(const MachineInstr &I) override {

    // TODO: relax this condition, if exec is masked, check that it goes back to
    // normal
    // TODO: what about scc or vcc ? Are they taken into acount in the MBB
    // live-ins ?
    MCRegister Exec = RegInfo->getExec();
    bool ModifiesExec = I.modifiesRegister(Exec, RegInfo);
    if (ModifiesExec)
      return false;

    if (needsExecPredication(TII, I))
      return true;

    bool DontMoveAcrossStore = true;
    bool IsSpeculatable = I.isDereferenceableInvariantLoad() ||
                          I.isSafeToMove(DontMoveAcrossStore);
    if (IsSpeculatable)
      return true;

    return false;
  }

  bool shouldConvertIf(SSAIfConv &IfConv) override {
    // TODO: cost model
    return true;
  }

  void predicateBlock(MachineBasicBlock *MBB, ArrayRef<MachineOperand> Cond,
                      bool Reverse) override {
    // save exec
    MachineFunction &MF = *MBB->getParent();
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

    Register ExecBackup = MFI->getSGPRForEXECCopy();

    const DebugLoc &CmpLoc = Cmp->getDebugLoc();

    auto FirstInstruction = MBB->begin();
    const bool IsSCCLive =
        false; // asume not since the live-ins are supposed to be empty
    TII->insertScratchExecCopy(MF, *MBB, FirstInstruction, CmpLoc, ExecBackup,
                               IsSCCLive);

    // mask exec
    unsigned CmpOpc = Cmp->getOpcode();
    if (TII->isSALU(*Cmp))
      CmpOpc = TII->getVALUOp(*Cmp);

    CmpOpc = AMDGPU::getVCMPXOpFromVCMP(CmpOpc);
    if (Reverse)
      CmpOpc = getReversedVCMPXOpcode(CmpOpc);

    // TODO: handle this properly. The second block may kill those registers.
    Cmp->getOperand(0).setIsKill(false);
    Cmp->getOperand(1).setIsKill(false);

    auto VCmpX = BuildMI(*MBB, FirstInstruction, CmpLoc, TII->get(CmpOpc));
    VCmpX->addOperand(Cmp->getOperand(0));
    VCmpX->addOperand(Cmp->getOperand(1));

    // restore exec
    TII->restoreExec(MF, *MBB, MBB->end(), DebugLoc(), ExecBackup);
  }

  ~ExecPredicate() override = default;
};

const char PassName[] = "AMDGPU If Conversion";

struct AMDGPUIfConverter : MachineFunctionPass {
  static char ID;
  AMDGPUIfConverter() : MachineFunctionPass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return PassName; }
};

char AMDGPUIfConverter::ID = 0;

void AMDGPUIfConverter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool AMDGPUIfConverter::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const auto &STI = MF.getSubtarget<GCNSubtarget>();
  if (!STI.hasGFX10_3Insts())
    return false;

  const SIInstrInfo *TII = STI.getInstrInfo();
  auto *DomTree = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto *Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  ExecPredicate Predicate(TII);
  SSAIfConv IfConv(Predicate, MF, DomTree, Loops);
  return IfConv.run();
}
} // namespace
char &llvm::AMDGPUIfConverterID = AMDGPUIfConverter::ID;
INITIALIZE_PASS_BEGIN(AMDGPUIfConverter, DEBUG_TYPE, PassName, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUIfConverter, DEBUG_TYPE, PassName, false, false)