//===-- PythonBindings.td - Python binding type mappings ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// C++ type and attribute to Python type mappings for -gen-python-op-bindings.
// Dialects can include this file and add their own mappings.
//
//===----------------------------------------------------------------------===//

#ifndef PYTHON_BINDINGS_TD
#define PYTHON_BINDINGS_TD

/// Maps a C++ type to a Python type annotation for operands and results.
/// Example: PythonTypeName<"::mlir::IntegerType", "_ods_ir.IntegerType">
class PythonTypeName<string cppType, string pythonType> {
  string cppName = cppType;
  string pyName = pythonType;
}

/// Maps a TableGen attribute def name to the Python type accepted by its
/// AttrBuilder. Example: PythonAttrType<"I32Attr", "int">
class PythonAttrType<string attrDefName, string pythonType> {
  string defName = attrDefName;
  string pyType = pythonType;
}

#endif // PYTHON_BINDINGS_TD
