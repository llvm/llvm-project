//===-- LLVMOpBase.td - LLVM IR dialect shared definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

// Case of the LLVM enum attribute backed by I64Attr with customized string
// representation that corresponds to what is visible in the textual IR form.
// The parameters are as follows:
//   - `cppSym`: name of the C++ enumerant for this case in MLIR API;
//   - `irSym`: keyword used in the custom form of MLIR operation;
//   - `llvmSym`: name of the C++ enumerant for this case in LLVM API.
// For example, `LLVM_EnumAttrCase<"Weak", "weak", "WeakAnyLinkage">` is usable
// as `<MlirEnumName>::Weak` in MLIR API, `WeakAnyLinkage` in LLVM API and
// is printed/parsed as `weak` in MLIR custom textual format.
class LLVM_EnumAttrCase<string cppSym, string irSym, string llvmSym, int val> :
    I64EnumAttrCase<cppSym, val, irSym> {

  // The name of the equivalent enumerant in LLVM.
  string llvmEnumerant = llvmSym;
}

// LLVM enum attribute backed by I64Attr with string representation
// corresponding to what is visible in the textual IR form.
// The parameters are as follows:
//   - `name`: name of the C++ enum class in MLIR API;
//   - `llvmName`: name of the C++ enum in LLVM API;
//   - `description`: textual description for documentation purposes;
//   - `cases`: list of enum cases.
// For example, `LLVM_EnumAttr<Linkage, "::llvm::GlobalValue::LinkageTypes`
// produces `mlir::LLVM::Linkage` enum class in MLIR API that corresponds to (a
// subset of) values in the `llvm::GlobalValue::LinkageTypes` in LLVM API.
class LLVM_EnumAttr<string name, string llvmName, string description,
                    list<LLVM_EnumAttrCase> cases> :
    I64EnumAttr<name, description, cases> {

  // The equivalent enum class name in LLVM.
  string llvmClassName = llvmName;
}

// LLVM vector reduction over a single vector.
class LLVM_VectorReduction<string mnem> :
  LLVM_IntrOp<"experimental.vector.reduce." # mnem, []>,
  Arguments<(ins LLVM_Type)>, Results<(outs LLVM_Type:$res)> {
    let llvmBuilder = [{
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::experimental_vector_reduce_}] #
        !subst(".","_", mnem) # [{, {
          opInst.getOperand(0).getType().cast<LLVM::LLVMType>()
          .getUnderlyingType(),
        });
      auto operands = lookupValues(opInst.getOperands());
      $res = builder.CreateCall(fn, operands);
    }];
}

// LLVM vector reduction over a single vector, with an initial value.
class LLVM_VectorReductionV2<string mnem> :
  LLVM_IntrOp<"experimental.vector.reduce.v2." # mnem, []>,
  Arguments<(ins LLVM_Type, LLVM_Type)>, Results<(outs LLVM_Type:$res)> {
    let llvmBuilder = [{
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::experimental_vector_reduce_v2_}] #
        !subst(".","_", mnem) # [{, {
          opInst.getResult(0).getType().cast<LLVM::LLVMType>()
          .getUnderlyingType(),
          opInst.getOperand(1).getType().cast<LLVM::LLVMType>()
          .getUnderlyingType(),
        });
      auto operands = lookupValues(opInst.getOperands());
      $res = builder.CreateCall(fn, operands);
    }];
}

#endif  // LLVMIR_OP_BASE
