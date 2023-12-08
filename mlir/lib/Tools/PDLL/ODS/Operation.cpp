//===- Operation.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/ODS/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pdll::ods;

//===----------------------------------------------------------------------===//
// Operation
//===----------------------------------------------------------------------===//

Operation::Operation(StringRef name, StringRef summary, StringRef desc,
                     StringRef nativeClassName, bool supportsTypeInferrence,
                     llvm::SMLoc loc)
    : name(name.str()), summary(summary.str()), description(desc.str()),
      nativeClassName(nativeClassName.str()),
      supportsTypeInferrence(supportsTypeInferrence),
      location(loc, llvm::SMLoc::getFromPointer(loc.getPointer() + 1)) {}
