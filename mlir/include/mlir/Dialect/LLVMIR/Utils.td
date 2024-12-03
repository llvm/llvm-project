//===-- Utils.td - MLIR LLVM IR utilities file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utilities to map from MLIR LLVM IR dialect to LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_UTILS_TD
#define MLIR_DIALECT_LLVMIR_UTILS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Base classes for LLVM enum attributes.
//===----------------------------------------------------------------------===//

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
//   - `cases`: list of enum cases;
//   - `unsupportedCases`: optional list of unsupported enum cases.
// For example, `LLVM_EnumAttr<Linkage, "::llvm::GlobalValue::LinkageTypes`
// produces `mlir::LLVM::Linkage` enum class in MLIR API that corresponds to (a
// subset of) values in the `llvm::GlobalValue::LinkageTypes` in LLVM API.
// All unsupported cases are excluded from the MLIR enum and trigger an error
// during the import from LLVM IR. They are useful to handle sentinel values
// such as `llvm::AtomicRMWInst::BinOp::BAD_BINOP` that LLVM commonly uses to
// terminate its enums.
class LLVM_EnumAttr<string name, string llvmName, string description,
                    list<LLVM_EnumAttrCase> cases,
                    list<LLVM_EnumAttrCase> unsupportedCases = []> :
    I64EnumAttr<name, description, cases> {
  // List of unsupported cases that have no conversion to an MLIR value.
  list<LLVM_EnumAttrCase> unsupported = unsupportedCases;

  // The equivalent enum class name in LLVM.
  string llvmClassName = llvmName;
}

// LLVM_CEnumAttr is functionally identical to LLVM_EnumAttr, but to be used for
// non-class enums.
class LLVM_CEnumAttr<string name, string llvmNS, string description,
      list<LLVM_EnumAttrCase> cases> :
    I64EnumAttr<name, description, cases> {
  string llvmClassName = llvmNS;
}

#endif // MLIR_DIALECT_LLVMIR_UTILS_TD
