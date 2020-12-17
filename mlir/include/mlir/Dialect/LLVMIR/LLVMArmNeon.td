//===-- LLVMArmNeon.td - LLVMArmNeon dialect op definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the LLVMArmNeon dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_ARMNEON_OPS
#define LLVMIR_ARMNEON_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// LLVMArmNeon dialect definition
//===----------------------------------------------------------------------===//

def LLVMArmNeon_Dialect : Dialect {
  let name = "llvm_arm_neon";
  let cppNamespace = "::mlir::LLVM";
}

//----------------------------------------------------------------------------//
// MLIR LLVMArmNeon intrinsics using the MLIR LLVM Dialect type system
//----------------------------------------------------------------------------//

class LLVMArmNeon_IntrBinaryOverloadedOp<string mnemonic, list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/LLVMArmNeon_Dialect,
                  /*string opName=*/mnemonic,
                  /*string enumName=*/"aarch64_neon_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[], // defined by result overload
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/1>;

def LLVM_aarch64_arm_neon_smull :
  LLVMArmNeon_IntrBinaryOverloadedOp<"smull">, Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // ARMNEON_OPS
