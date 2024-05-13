//===- Constraint.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/ODS/Constraint.h"

using namespace mlir;
using namespace mlir::pdll::ods;

//===----------------------------------------------------------------------===//
// Constraint
//===----------------------------------------------------------------------===//

StringRef Constraint::getDemangledName() const {
  StringRef demangledName = name;

  // Drop the "anonymous" suffix if present.
  size_t anonymousSuffix = demangledName.find("(anonymous_");
  if (anonymousSuffix != StringRef::npos)
    demangledName = demangledName.take_front(anonymousSuffix);
  return demangledName;
}
