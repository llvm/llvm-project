//===- Builder.cpp - Builder definitions ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Builder.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Builder::Parameter
//===----------------------------------------------------------------------===//

/// Return a string containing the C++ type of this parameter.
StringRef Builder::Parameter::getCppType() const {
  if (const auto *stringInit = dyn_cast<llvm::StringInit>(def))
    return stringInit->getValue();
  const llvm::Record *record = cast<llvm::DefInit>(def)->getDef();
  return record->getValueAsString("type");
}

/// Return an optional string containing the default value to use for this
/// parameter.
Optional<StringRef> Builder::Parameter::getDefaultValue() const {
  if (isa<llvm::StringInit>(def))
    return llvm::None;
  const llvm::Record *record = cast<llvm::DefInit>(def)->getDef();
  Optional<StringRef> value = record->getValueAsOptionalString("defaultValue");
  return value && !value->empty() ? value : llvm::None;
}

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

Builder::Builder(const llvm::Record *record, ArrayRef<llvm::SMLoc> loc)
    : def(record) {
  // Initialize the parameters of the builder.
  const llvm::DagInit *dag = def->getValueAsDag("dagParams");
  auto *defInit = dyn_cast<llvm::DefInit>(dag->getOperator());
  if (!defInit || !defInit->getDef()->getName().equals("ins"))
    PrintFatalError(def->getLoc(), "expected 'ins' in builders");

  bool seenDefaultValue = false;
  for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i) {
    const llvm::StringInit *paramName = dag->getArgName(i);
    const llvm::Init *paramValue = dag->getArg(i);
    Parameter param(paramName ? paramName->getValue() : Optional<StringRef>(),
                    paramValue);

    // Similarly to C++, once an argument with a default value is detected, the
    // following arguments must have default values as well.
    if (param.getDefaultValue()) {
      seenDefaultValue = true;
    } else if (seenDefaultValue) {
      PrintFatalError(loc,
                      "expected an argument with default value after other "
                      "arguments with default values");
    }
    parameters.emplace_back(param);
  }
}

/// Return an optional string containing the body of the builder.
Optional<StringRef> Builder::getBody() const {
  Optional<StringRef> body = def->getValueAsOptionalString("body");
  return body && !body->empty() ? body : llvm::None;
}
