//===-- LLVMOpBase.td - LLVM IR dialect shared definitions -*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains shared definitions for the LLVM IR dialect and its
// subdialects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_OP_BASE
#define LLVMIR_OP_BASE

include "mlir/IR/OpBase.td"

def LLVM_Dialect : Dialect {
  let name = "llvm";
  let cppNamespace = "LLVM";
}

// LLVM IR type wrapped in MLIR.
def LLVM_Type : Type<CPred<"$_self.isa<::mlir::LLVM::LLVMType>()">,
                     "LLVM dialect type">;

// Type constraint accepting only wrapped LLVM integer types.
def LLVMInt : TypeConstraint<
  And<[LLVM_Type.predicate,
       CPred<"$_self.cast<::mlir::LLVM::LLVMType>().isIntegerTy()">]>,
  "LLVM dialect integer">;

// Base class for LLVM operations. Defines the interface to the llvm::IRBuilder
// used to translate to LLVM IR proper.
class LLVM_OpBase<Dialect dialect, string mnemonic, list<OpTrait> traits = []> :
    Op<dialect, mnemonic, traits> {
  // A pattern for constructing the LLVM IR Instruction (or other Value) that
  // corresponds to this op.  This pattern can use `builder` to refer to an
  // `llvm::IRBuilder<>` instance, $-names of arguments and results and the
  // following special variable names:
  //   - $_resultType - substituted with the LLVM IR type of the result;
  //   - $_numOperands - substituted with the number of operands (including
  //                     the variadic ones);
  //   - $_hasResult - substituted with a check that a variadic-result op does
  //                   have a result (LLVM ops can have 0 or 1 result);
  //   - $_location - mlir::Location object of the instruction.
  // Additionally, `$$` can be used to produce the dollar character.
  string llvmBuilder = "";
}

// Base class for LLVM operations. All operations get an "llvm." prefix in
// their name automatically. LLVM operations have either zero or one result,
// this class is specialized below for both cases and should not be used
// directly.
class LLVM_Op<string mnemonic, list<OpTrait> traits = []> :
    LLVM_OpBase<LLVM_Dialect, mnemonic, traits>;

// Base class for LLVM intrinsics operation. It is the same as an LLVM_Op
// but the operation has a ".intr." element in the prefix becoming
// "llvm.intr.*".
class LLVM_IntrOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<"intr."#mnemonic, traits>;

#endif  // LLVMIR_OP_BASE
