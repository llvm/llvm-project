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

#include "mlir/TableGen/Generators/OpGenHelpers.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Error.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

static std::string getOperationName(const Record &def) {
  auto prefix = def.getValueAsDef("opDialect")->getValueAsString("name");
  auto opName = def.getValueAsString("opName");
  if (prefix.empty())
    return std::string(opName);
  return std::string(formatv("{0}.{1}", prefix, opName));
}

std::vector<const Record *>
mlir::tblgen::getRequestedOpDefinitions(const RecordKeeper &records,
                                        StringRef includeRegex,
                                        StringRef excludeRegex) {
  const Record *classDef = records.getClass("Op");
  if (!classDef)
    PrintFatalError("ERROR: Couldn't find the 'Op' class!\n");

  Regex incRegex(includeRegex), excRegex(excludeRegex);
  std::vector<const Record *> defs;
  for (const auto &def : records.getDefs()) {
    if (!def.second->isSubClassOf(classDef))
      continue;
    if (!includeRegex.empty() && !incRegex.match(getOperationName(*def.second)))
      continue;
    if (!excludeRegex.empty() && excRegex.match(getOperationName(*def.second)))
      continue;
    defs.push_back(def.second.get());
  }

  return defs;
}

bool mlir::tblgen::isPythonReserved(StringRef str) {
  static StringSet<> reserved({
      "False",  "None",   "True",    "and",      "as",       "assert", "async",
      "await",  "break",  "class",   "continue", "def",      "del",    "elif",
      "else",   "except", "finally", "for",      "from",     "global", "if",
      "import", "in",     "is",      "lambda",   "nonlocal", "not",    "or",
      "pass",   "raise",  "return",  "try",      "while",    "with",   "yield",
  });
  reserved.insert("callable");
  reserved.insert("issubclass");
  reserved.insert("type");
  return reserved.contains(str);
}

void mlir::tblgen::shardOpDefinitions(
    ArrayRef<const Record *> defs,
    SmallVectorImpl<ArrayRef<const Record *>> &shardedDefs,
    unsigned shardCount) {
  assert(shardCount > 0 && "expected a positive shard count");
  if (shardCount == 1) {
    shardedDefs.push_back(defs);
    return;
  }

  unsigned minShardSize = defs.size() / shardCount;
  unsigned numMissing = defs.size() - minShardSize * shardCount;
  shardedDefs.reserve(shardCount);
  for (unsigned i = 0, start = 0; i < shardCount; ++i) {
    unsigned size = minShardSize + (i < numMissing);
    shardedDefs.push_back(defs.slice(start, size));
    start += size;
  }
}
