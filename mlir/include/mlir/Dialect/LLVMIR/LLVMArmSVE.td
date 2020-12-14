//===-- LLVMArmSVE.td - LLVMARMSVE dialect op definitions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the LLVMArmSVE dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_ARMSVE_OPS
#define LLVMIR_ARMSVE_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

//===----------------------------------------------------------------------===//
// LLVMArmSVE dialect definition
//===----------------------------------------------------------------------===//

def LLVMArmSVE_Dialect : Dialect {
  let name = "llvm_arm_sve";
  let cppNamespace = "::mlir::LLVM";
}

//----------------------------------------------------------------------------//
// MLIR LLVM Arm SVE intrinsics using the MLIR LLVM Dialect type system
//----------------------------------------------------------------------------//

class LLVMArmSVE_NonSVEIntrUnaryOverloadedOp<string mnemonic,
                                             list<OpTrait> traits =[]> :
  LLVM_IntrOpBase</*Dialect dialect=*/LLVMArmSVE_Dialect,
                  /*string opName=*/mnemonic,
                  /*string enumName=*/mnemonic,
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[], // defined by result overload
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/1>;

class LLVMArmSVE_IntrBinaryOverloadedOp<string mnemonic,
                                        list<OpTrait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/LLVMArmSVE_Dialect,
                  /*string opName=*/mnemonic,
                  /*string enumName=*/"aarch64_sve_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0],
                  /*list<int> overloadedOperands=*/[], // defined by result overload
                  /*list<OpTrait> traits=*/traits,
                  /*int numResults=*/1>;

def LLVM_aarch64_arm_sve_ummla :
  LLVMArmSVE_IntrBinaryOverloadedOp<"ummla">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_aarch64_arm_sve_smmla :
  LLVMArmSVE_IntrBinaryOverloadedOp<"smmla">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_aarch64_arm_sve_sdot :
  LLVMArmSVE_IntrBinaryOverloadedOp<"sdot">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_aarch64_arm_sve_udot :
  LLVMArmSVE_IntrBinaryOverloadedOp<"udot">,
  Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

def LLVM_vector_scale :
  LLVMArmSVE_NonSVEIntrUnaryOverloadedOp<"vscale">;

#endif // ARMSVE_OPS
