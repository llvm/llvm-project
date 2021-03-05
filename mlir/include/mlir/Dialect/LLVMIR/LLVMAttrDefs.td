//===-- LLVMAttrDefs.td - LLVM Attributes definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_ATTRDEFS
#define LLVMIR_ATTRDEFS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"


// All of the attributes will extend this class.
class LLVM_Attr<string name> : AttrDef<LLVM_Dialect, name>;

// The "FastMath" flags associated with floating point LLVM instructions.
def FastmathFlagsAttr : LLVM_Attr<"FMF"> {
  let mnemonic = "fastmath";

  // List of type parameters.
  let parameters = (
    ins
    "FastmathFlags":$flags
  );
}

#endif // LLVMIR_ATTRDEFS
