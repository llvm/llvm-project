//===- Dialect.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/ODS/Dialect.h"
#include "mlir/Tools/PDLL/ODS/Constraint.h"
#include "mlir/Tools/PDLL/ODS/Operation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pdll::ods;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

Dialect::Dialect(StringRef name) : name(name.str()) {}
Dialect::~Dialect() = default;

std::pair<Operation *, bool>
Dialect::insertOperation(StringRef name, StringRef summary, StringRef desc,
                         StringRef nativeClassName,
                         bool supportsResultTypeInferrence, llvm::SMLoc loc) {
  std::unique_ptr<Operation> &operation = operations[name];
  if (operation)
    return std::make_pair(&*operation, /*wasInserted*/ false);

  operation.reset(new Operation(name, summary, desc, nativeClassName,
                                supportsResultTypeInferrence, loc));
  return std::make_pair(&*operation, /*wasInserted*/ true);
}

Operation *Dialect::lookupOperation(StringRef name) const {
  auto it = operations.find(name);
  return it != operations.end() ? it->second.get() : nullptr;
}
