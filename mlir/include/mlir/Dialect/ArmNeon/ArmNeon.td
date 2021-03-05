//===-- ArmNeonOps.td - ArmNeon dialect op definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the ArmNeon dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ARMNEON_OPS
#define ARMNEON_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ArmNeon dialect definition
//===----------------------------------------------------------------------===//

def ArmNeon_Dialect : Dialect {
  let name = "arm_neon";
  let cppNamespace = "::mlir::arm_neon";

  // Note: this does not need to depend on LLVMDialect as long as functions in
  // this dialect (such as canonicalization) do not produce entities belonging
  // to the LLVMDialect (ops or types).
}

//===----------------------------------------------------------------------===//
// ArmNeon op definitions
//===----------------------------------------------------------------------===//

// ArmNeon dialect op that corresponds (and is convertible to) an LLVM IR
// intrinsic.
class ArmNeon_IntrOp<string mnemonic, list<int> overloadedResults,
                     list<int> overloadedOperands, int numResults,
                     list<OpTrait> traits = [], bit requiresAccessGroup = 0>
    : LLVM_IntrOpBase</*dialect=*/ArmNeon_Dialect,
                      /*opName=*/mnemonic,
                      /*enumName=*/"aarch64_neon_" # !subst(".", "_", mnemonic),
                      /*overloadedResults=*/overloadedResults,
                      /*overloadedOperands=*/overloadedOperands,
                      /*traits=*/traits,
                      /*numResults=*/numResults,
                      /*requiresAccessGroup=*/requiresAccessGroup>;

// ArmNeon dialect op that corresponds to an LLVM IR intrinsic with one
// overloaded result.
class ArmNeon_OverloadedOneResultIntrOp<string mnemonic,
                                        list<OpTrait> traits = []>
  : ArmNeon_IntrOp<mnemonic, [0], [], 1, traits>;

def SMullOp : ArmNeon_OverloadedOneResultIntrOp<"smull", [
       NoSideEffect,
       AllTypesMatch<["a", "b"]>,
       TypesMatchWith<
         "res has same vector shape and element bitwidth scaled by 2 as a",
         "a", "res", "$_self.cast<VectorType>().scaleElementBitwidth(2)">
    ]> {
  let summary = "smull roundscale op";
  let description = [{
    Signed Multiply Long (vector). This instruction multiplies corresponding
    signed integer values in the lower or upper half of the vectors of the two
    source SIMD&FP registers, places the results in a vector, and writes the
    vector to the destination SIMD&FP register.

    Source:
    https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
  }];

  // Supports either:
  //   (vector<8xi8>, vector<8xi8>) -> (vector<8xi16>)
  //   (vector<4xi16>, vector<4xi16>) -> (vector<4xi32>)
  //   (vector<2xi32>, vector<2xi32>) -> (vector<2xi64>)
  let arguments = (ins VectorOfLengthAndType<[8, 4, 2], [I8, I16, I32]>:$a,
                       VectorOfLengthAndType<[8, 4, 2], [I8, I16, I32]>:$b);
  let results = (outs VectorOfLengthAndType<[8, 4, 2], [I16, I32, I64]>:$res);
  let assemblyFormat =
    "$a `,` $b attr-dict `:` type($a) `to` type($res)";
}


#endif // ARMNEON_OPS
