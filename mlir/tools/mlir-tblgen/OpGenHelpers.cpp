//===- OpGenHelpers.cpp - MLIR operation generator helpers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers used in the op generators.
//
//===----------------------------------------------------------------------===//

#include "OpGenHelpers.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Error.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

cl::OptionCategory opDefGenCat("Options for op definition generators");

static cl::opt<std::string> opIncFilter(
    "op-include-regex",
    cl::desc("Regex of name of op's to include (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<std::string> opExcFilter(
    "op-exclude-regex",
    cl::desc("Regex of name of op's to exclude (no filter if empty)"),
    cl::cat(opDefGenCat));

static std::string getOperationName(const Record &def) {
  auto prefix = def.getValueAsDef("opDialect")->getValueAsString("name");
  auto opName = def.getValueAsString("opName");
  if (prefix.empty())
    return std::string(opName);
  return std::string(llvm::formatv("{0}.{1}", prefix, opName));
}

std::vector<Record *>
mlir::tblgen::getRequestedOpDefinitions(const RecordKeeper &recordKeeper) {
  Record *classDef = recordKeeper.getClass("Op");
  if (!classDef)
    PrintFatalError("ERROR: Couldn't find the 'Op' class!\n");

  llvm::Regex includeRegex(opIncFilter), excludeRegex(opExcFilter);
  std::vector<Record *> defs;
  for (const auto &def : recordKeeper.getDefs()) {
    if (!def.second->isSubClassOf(classDef))
      continue;
    // Include if no include filter or include filter matches.
    if (!opIncFilter.empty() &&
        !includeRegex.match(getOperationName(*def.second)))
      continue;
    // Unless there is an exclude filter and it matches.
    if (!opExcFilter.empty() &&
        excludeRegex.match(getOperationName(*def.second)))
      continue;
    defs.push_back(def.second.get());
  }

  return defs;
}
