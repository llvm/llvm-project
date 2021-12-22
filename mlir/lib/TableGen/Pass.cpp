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
// PassOption
//===----------------------------------------------------------------------===//

StringRef PassOption::getCppVariableName() const {
  return def->getValueAsString("cppName");
}

StringRef PassOption::getArgument() const {
  return def->getValueAsString("argument");
}

StringRef PassOption::getType() const { return def->getValueAsString("type"); }

Optional<StringRef> PassOption::getDefaultValue() const {
  StringRef defaultVal = def->getValueAsString("defaultValue");
  return defaultVal.empty() ? Optional<StringRef>() : defaultVal;
}

StringRef PassOption::getDescription() const {
  return def->getValueAsString("description");
}

Optional<StringRef> PassOption::getAdditionalFlags() const {
  StringRef additionalFlags = def->getValueAsString("additionalOptFlags");
  return additionalFlags.empty() ? Optional<StringRef>() : additionalFlags;
}

bool PassOption::isListOption() const {
  return def->isSubClassOf("ListOption");
}

//===----------------------------------------------------------------------===//
// PassStatistic
//===----------------------------------------------------------------------===//

StringRef PassStatistic::getCppVariableName() const {
  return def->getValueAsString("cppName");
}

StringRef PassStatistic::getName() const {
  return def->getValueAsString("name");
}

StringRef PassStatistic::getDescription() const {
  return def->getValueAsString("description");
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

Pass::Pass(const llvm::Record *def) : def(def) {
  for (auto *init : def->getValueAsListOfDefs("options"))
    options.emplace_back(init);
  for (auto *init : def->getValueAsListOfDefs("statistics"))
    statistics.emplace_back(init);
  for (StringRef dialect : def->getValueAsListOfStrings("dependentDialects"))
    dependentDialects.push_back(dialect);
}

StringRef Pass::getArgument() const {
  return def->getValueAsString("argument");
}

StringRef Pass::getBaseClass() const {
  return def->getValueAsString("baseClass");
}

StringRef Pass::getSummary() const { return def->getValueAsString("summary"); }

StringRef Pass::getDescription() const {
  return def->getValueAsString("description");
}

StringRef Pass::getConstructor() const {
  return def->getValueAsString("constructor");
}
ArrayRef<StringRef> Pass::getDependentDialects() const {
  return dependentDialects;
}

ArrayRef<PassOption> Pass::getOptions() const { return options; }

ArrayRef<PassStatistic> Pass::getStatistics() const { return statistics; }
