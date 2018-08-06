//== ----- llvm/CodeGen/GlobalISel/Combiner.h --------------------- == //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// This contains common code to drive combines. Combiner Passes will need to
/// setup a CombinerInfo and call combineMachineFunction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_COMBINER_H
#define LLVM_CODEGEN_GLOBALISEL_COMBINER_H

#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
class MachineRegisterInfo;
class CombinerInfo;
class TargetPassConfig;
class MachineFunction;

class Combiner {
public:
  Combiner(CombinerInfo &CombinerInfo, const TargetPassConfig *TPC);

  bool combineMachineInstrs(MachineFunction &MF);

protected:
  CombinerInfo &CInfo;

  MachineRegisterInfo *MRI = nullptr;
  const TargetPassConfig *TPC;
  MachineIRBuilder Builder;
};

} // End namespace llvm.

#endif // LLVM_CODEGEN_GLOBALISEL_GICOMBINER_H
