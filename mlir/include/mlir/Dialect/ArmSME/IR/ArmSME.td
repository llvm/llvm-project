//===-- ArmSME.td - ArmSME dialect operation definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the ArmSME dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ArmSME
#define ArmSME

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// ArmSME dialect definition.
//===----------------------------------------------------------------------===//

def ArmSME_Dialect : Dialect {
  let name = "arm_sme";
  let cppNamespace = "::mlir::arm_sme";
  let summary = "Dialect to target the Armv9 Scalable Matrix Extension (SME)";
  let description = [{
    This dialect contains the definitions necessary to target specific Arm SME
    operations.

    For more details on the architecture, see the Arm documentation:
    https://developer.arm.com/documentation/ddi0616
  }];
  let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// LLVMIR Intrinsics
//===----------------------------------------------------------------------===//

class ArmSME_IntrOp<string mnemonic, int numResults = 1,
                    list<Trait> traits = []> :
  LLVM_IntrOpBase<ArmSME_Dialect, "intr." # mnemonic,
                  "aarch64_sme_" # !subst(".", "_", mnemonic), [], [], traits,
                  numResults>;

/// Create a call to aarch64_sme_zero intrinsic.
def LLVM_aarch64_sme_zero
    : ArmSME_IntrOp<"zero", 0>, Arguments<(ins I32:$imm)>;

#endif // ArmSME
