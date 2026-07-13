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
using llvm::DagInit;
using llvm::DefInit;
using llvm::Init;
using llvm::Record;
using llvm::StringInit;

//===----------------------------------------------------------------------===//
// Builder::Parameter
//===----------------------------------------------------------------------===//

/// Return a string containing the C++ type of this parameter.
StringRef Builder::Parameter::getCppType() const {
  if (const auto *stringInit = dyn_cast<StringInit>(def))
    return stringInit->getValue();
  const Record *record = cast<DefInit>(def)->getDef();
  // Inlining the first part of `Record::getValueAsString` to give better
  // error messages.
  const llvm::RecordVal *type = record->getValue("type");
  if (!type || !type->getValue()) {
    llvm::PrintFatalError("Builder DAG arguments must be either strings or "
                          "defs which inherit from CArg");
  }
  return record->getValueAsString("type");
}

/// Return an optional string containing the default value to use for this
/// parameter.
std::optional<StringRef> Builder::Parameter::getDefaultValue() const {
  if (isa<StringInit>(def))
    return std::nullopt;
  const Record *record = cast<DefInit>(def)->getDef();
  std::optional<StringRef> value =
      record->getValueAsOptionalString("defaultValue");
  return value && !value->empty() ? value : std::nullopt;
}

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

Builder::Builder(const Record *record, ArrayRef<SMLoc> loc) : def(record) {
  // Initialize the parameters of the builder.
  const DagInit *dag = def->getValueAsDag("dagParams");
  auto *defInit = dyn_cast<DefInit>(dag->getOperator());
  if (!defInit || defInit->getDef()->getName() != "ins")
    PrintFatalError(def->getLoc(), "expected 'ins' in builders");

  bool seenDefaultValue = false;
  for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i) {
    const StringInit *paramName = dag->getArgName(i);
    const Init *paramValue = dag->getArg(i);
    Parameter param(paramName ? paramName->getValue()
                              : std::optional<StringRef>(),
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
std::optional<StringRef> Builder::getBody() const {
  std::optional<StringRef> body = def->getValueAsOptionalString("body");
  return body && !body->empty() ? body : std::nullopt;
}

std::optional<StringRef> Builder::getDeprecatedMessage() const {
  std::optional<StringRef> message =
      def->getValueAsOptionalString("odsCppDeprecated");
  return message && !message->empty() ? message : std::nullopt;
}
