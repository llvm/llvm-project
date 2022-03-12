//== ----- llvm/CodeGen/GlobalISel/Combiner.h -------------------*- C++ -*-== //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This contains common code to drive combines. Combiner Passes will need to
/// setup a CombinerInfo and call combineMachineFunction.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_COMBINER_H
#define LLVM_CODEGEN_GLOBALISEL_COMBINER_H

#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

namespace llvm {
class MachineRegisterInfo;
class CombinerInfo;
class GISelCSEInfo;
class TargetPassConfig;
class MachineFunction;

class Combiner {
public:
  Combiner(CombinerInfo &CombinerInfo, const TargetPassConfig *TPC);

  /// If CSEInfo is not null, then the Combiner will setup observer for
  /// CSEInfo and instantiate a CSEMIRBuilder. Pass nullptr if CSE is not
  /// needed.
  bool combineMachineInstrs(MachineFunction &MF, GISelCSEInfo *CSEInfo);

protected:
  CombinerInfo &CInfo;

  MachineRegisterInfo *MRI = nullptr;
  const TargetPassConfig *TPC;
  std::unique_ptr<MachineIRBuilder> Builder;
};

} // End namespace llvm.

#endif // LLVM_CODEGEN_GLOBALISEL_COMBINER_H
