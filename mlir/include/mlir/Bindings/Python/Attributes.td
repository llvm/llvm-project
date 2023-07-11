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

#endif
