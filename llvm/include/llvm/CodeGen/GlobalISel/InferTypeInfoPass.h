#ifndef LLVM_CODEGEN_GLOBALISEL_INFERTYPEINFOPASS_H
#define LLVM_CODEGEN_GLOBALISEL_INFERTYPEINFOPASS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

class InferTypeInfo : public MachineFunctionPass {
public:
  static char ID;

private:
  MachineRegisterInfo *MRI = nullptr;
  MachineFunction *MF = nullptr;

  MachineIRBuilder Builder;

  /// Initialize the field members using \p MF.
  void init(MachineFunction &MF);

public:
  InferTypeInfo() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool inferTypeInfo(MachineFunction &MF);

  bool shouldBeFP(MachineOperand &Op, unsigned Depth) const;

  void updateDef(Register Reg);

  void updateUse(MachineOperand &Op, bool FP);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_INFERTYPEINFOPASS_H
