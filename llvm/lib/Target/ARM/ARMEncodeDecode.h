#ifndef ARM_ENCODE_DECODE
#define ARM_ENCODE_DECODE

#include "ARMRandezvousInstrumentor.h"
#include "llvm/Pass.h"
#include "llvm/Support/RandomNumberGenerator.h"

namespace llvm {
  struct ARMEncodeDecode : public ModulePass, ARMRandezvousInstrumentor {
    // Pass Identifier
    static char ID;

    // xor number
    static constexpr Register storeReg = ARM::R8;
    static constexpr Register XorReg = ARM::R9;
    static constexpr StringRef InitFuncName = "__xor_register_init";

    ARMEncodeDecode();
    virtual StringRef getPassName() const override;
    void getAnalysisUsage(AnalysisUsage & AU) const override;
    virtual bool runOnModule(Module & M) override;

  private:
    std::unique_ptr<RandomNumberGenerator> RNG;

    Function * createInitFunction(Module & M);
    bool EncodeLR(MachineInstr & MI, MachineOperand & LR,
                           uint32_t Stride);
    bool insertNop(MachineInstr &MI);
    bool EncodeCallSite(MachineInstr & MI, MachineOperand & MO,
                           uint32_t Stride);
    bool DecodeLR(MachineInstr & MI, MachineOperand & PCLR,
                            uint32_t Stride);
  };

  ModulePass * createARMEncodeDecode(void);
}

#endif