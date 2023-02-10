#ifndef ARM_RANDEZVOUS_SHADOW_STACK
#define ARM_RANDEZVOUS_SHADOW_STACK

#include "ARMRandezvousInstrumentor.h"
#include "llvm/Pass.h"
#include "llvm/Support/RandomNumberGenerator.h"

namespace llvm {
  struct ARMRandezvousShadowStack : public ModulePass, ARMRandezvousInstrumentor {
    // Pass Identifier
    static char ID;

    static constexpr Register ShadowStackPtrReg = ARM::R8;
    static constexpr Register ShadowStackStrideReg = ARM::R9;
    static constexpr StringRef ShadowStackName = "__randezvous_shadow_stack";
    static constexpr StringRef InitFuncName = "__randezvous_shadow_stack_init";

    ARMRandezvousShadowStack();
    virtual StringRef getPassName() const override;
    void getAnalysisUsage(AnalysisUsage & AU) const override;
    void releaseMemory() override;
    virtual bool runOnModule(Module & M) override;

  private:
    std::unique_ptr<RandomNumberGenerator> RNG;
    std::vector<MachineBasicBlock *> TrapBlocks;

    GlobalVariable * createShadowStack(Module & M);
    Function * createInitFunction(Module & M, GlobalVariable & SS);
    bool pushToShadowStack(MachineInstr & MI, MachineOperand & LR,
                           uint32_t Stride);
    bool popFromShadowStack(MachineInstr & MI, MachineOperand & PCLR,
                            uint32_t Stride);
    bool nullifyReturnAddress(MachineInstr & MI, MachineOperand & PCLR);
  };

  ModulePass * createARMRandezvousShadowStack(void);
}

#endif