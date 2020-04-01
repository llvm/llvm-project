//===- Pass.cpp - Pass related classes ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Pass.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

Pass::Pass(const llvm::Record *def) : def(def) {}

StringRef Pass::getArgument() const {
  return def->getValueAsString("argument");
}

StringRef Pass::getSummary() const { return def->getValueAsString("summary"); }

StringRef Pass::getDescription() const {
  return def->getValueAsString("description");
}

StringRef Pass::getConstructor() const {
  return def->getValueAsString("constructor");
}
