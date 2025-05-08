//===- ParasolInstructionSelector.cpp --------------------------------*- C++
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the InstructionSelector class for
/// Parasol.
//===----------------------------------------------------------------------===//

#include "ParasolInstrInfo.h"
#include "ParasolRegisterBankInfo.h"
#include "ParasolSubtarget.h"
#include "ParasolTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "parasol-gisel"

using namespace llvm;

namespace {

#define GET_GLOBALISEL_PREDICATE_BITSET
#include "ParasolGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATE_BITSET

class ParasolInstructionSelector : public InstructionSelector {
public:
  ParasolInstructionSelector(const ParasolTargetMachine &TM,
                             const ParasolSubtarget &STI,
                             const ParasolRegisterBankInfo &RBI);

  bool select(MachineInstr &I) override;
  static const char *getName() { return DEBUG_TYPE; }

private:
  bool preISelLower(MachineInstr &I);

  /// tblgen generated 'select' implementation that is used as the initial
  /// selector for the patterns that do not require complex C++.
  bool selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const;

  const ParasolInstrInfo &TII;
  const ParasolRegisterInfo &TRI;
  const ParasolRegisterBankInfo &RBI;

#define GET_GLOBALISEL_PREDICATES_DECL
#include "ParasolGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATES_DECL

#define GET_GLOBALISEL_TEMPORARIES_DECL
#include "ParasolGenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_DECL
};

} // namespace

#define GET_GLOBALISEL_IMPL
#include "ParasolGenGlobalISel.inc"
#undef GET_GLOBALISEL_IMPL

ParasolInstructionSelector::ParasolInstructionSelector(
    const ParasolTargetMachine &TM, const ParasolSubtarget &STI,
    const ParasolRegisterBankInfo &RBI)
    : TII(*STI.getInstrInfo()), TRI(*STI.getRegisterInfo()), RBI(RBI),
#define GET_GLOBALISEL_PREDICATES_INIT
#include "ParasolGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATES_INIT
#define GET_GLOBALISEL_TEMPORARIES_INIT
#include "ParasolGenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_INIT
{
}

bool ParasolInstructionSelector::preISelLower(MachineInstr &I) {
  MachineBasicBlock &MBB = *I.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // SUNSCREEN TODO: Figure out which machine instructions should be selected
  // here.
  return false;
}

bool ParasolInstructionSelector::select(MachineInstr &I) {
  if (preISelLower(I)) {
    // Opcode = I.getOpcode(); // The opcode may have been modified, refresh it.
  }

  if (!isPreISelGenericOpcode(I.getOpcode()))
    return true;
  if (selectImpl(I, *CoverageInfo))
    return true;
  return false;
}

namespace llvm {
InstructionSelector *
createParasolInstructionSelector(const ParasolTargetMachine &TM,
                                 const ParasolSubtarget &Subtarget,
                                 const ParasolRegisterBankInfo &RBI) {
  return new ParasolInstructionSelector(TM, Subtarget, RBI);
}
} // namespace llvm
