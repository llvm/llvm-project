#ifndef ARM_TRAMPOLINE
#define ARM_TRAMPOLINE

#include "ARMRandezvousInstrumentor.h"
#include "llvm/Pass.h"
#include "llvm/Support/RandomNumberGenerator.h"

namespace llvm {
  struct ARMTrampoline : public ModulePass, ARMRandezvousInstrumentor {
    // Pass Identifier
    static char ID;

    ARMTrampoline();
    virtual StringRef getPassName() const override;
    void getAnalysisUsage(AnalysisUsage & AU) const override;
    virtual bool runOnModule(Module & M) override;

  private:
    bool insertNop(MachineInstr &MI);
    bool BlxTrampoline(MachineInstr &MI, MachineOperand &MO);
  };

  ModulePass * createARMTrampoline(void);
}

#endif