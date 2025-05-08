//=== Parasol.h - Top-level interface for Parasol representation ----------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM Parasol backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_PARASOL_H
#define LLVM_LIB_TARGET_PARASOL_PARASOL_H

#include "MCTargetDesc/ParasolMCTargetDesc.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class FunctionPass;
class InstructionSelector;
class ParasolRegisterBankInfo;
class ParasolSubtarget;
class ParasolTargetMachine;
class PassRegistry;

// Declare functions to create passes here!
FunctionPass *createAnnotateEncryptionPass();
void initializeAnnotateEncryptionLegacyPass(PassRegistry &);

InstructionSelector *
createParasolInstructionSelector(const ParasolTargetMachine &,
                                 const ParasolSubtarget &,
                                 const ParasolRegisterBankInfo &);

} // namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_PARASOL_H
