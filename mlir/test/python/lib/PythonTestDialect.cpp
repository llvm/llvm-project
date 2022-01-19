//===- PythonTestDialect.cpp - PythonTest dialect definition --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PythonTestDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PythonTestDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "PythonTestAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PythonTestTypes.cpp.inc"

#define GET_OP_CLASSES
#include "PythonTestOps.cpp.inc"

namespace python_test {
void PythonTestDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PythonTestOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "PythonTestAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "PythonTestTypes.cpp.inc"
      >();
}

} // namespace python_test
