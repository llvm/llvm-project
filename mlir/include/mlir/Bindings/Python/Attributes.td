//===-- Attributes.td - Attribute mapping for Python -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the mapping between MLIR ODS attributes and the corresponding
// Python binding classes.
//
//===----------------------------------------------------------------------===//

#ifndef PYTHON_BINDINGS_ATTRIBUTES
#define PYTHON_BINDINGS_ATTRIBUTES

// A mapping between the attribute storage type and the corresponding Python
// type. There is not necessarily a 1-1 match for non-builtin attributes.
class PythonAttr<string c, string p> {
  string cppStorageType = c;
  string pythonType = p;
}

// Mappings between supported builtin attribtues and Python types.
def : PythonAttr<"::mlir::Attribute", "_ods_ir.Attribute">;
def : PythonAttr<"::mlir::BoolAttr", "_ods_ir.BoolAttr">;
def : PythonAttr<"::mlir::IntegerAttr", "_ods_ir.IntegerAttr">;
def : PythonAttr<"::mlir::FloatAttr", "_ods_ir.FloatAttr">;
def : PythonAttr<"::mlir::StringAttr", "_ods_ir.StringAttr">;
def : PythonAttr<"::mlir::DenseElementsAttr", "_ods_ir.DenseElementsAttr">;
def : PythonAttr<"::mlir::DenseIntElementsAttr", "_ods_ir.DenseIntElementsAttr">;
def : PythonAttr<"::mlir::DenseFPElementsAttr", "_ods_ir.DenseFPElementsAttr">;

#endif
