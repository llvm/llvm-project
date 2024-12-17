//===- MlirTblgenMain.cpp - MLIR Tablegen Driver main -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-tblgen for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace llvm;

enum DeprecatedAction { None, Warn, Error };

static DeprecatedAction actionOnDeprecatedValue;

// Returns if there is a use of `deprecatedInit` in `field`.
static bool findUse(const Init *field, const Init *deprecatedInit,
                    llvm::DenseMap<const Init *, bool> &known) {
  if (field == deprecatedInit)
    return true;

  auto it = known.find(field);
  if (it != known.end())
    return it->second;

  auto memoize = [&](bool val) {
    known[field] = val;
    return val;
  };

  if (auto *defInit = dyn_cast<DefInit>(field)) {
    // Only recurse into defs if they are anonymous.
    // Non-anonymous defs are handled by the main loop, with a proper
    // deprecation warning for each. Returning true here, would cause
    // all users of a def to also emit a deprecation warning.
    if (!defInit->getDef()->isAnonymous())
      // Purposefully not memoize as to not include every def use in the map.
      // This is also a trivial case we return false for in constant time.
      return false;

    return memoize(
        llvm::any_of(defInit->getDef()->getValues(), [&](const RecordVal &val) {
          return findUse(val.getValue(), deprecatedInit, known);
        }));
  }

  if (auto *dagInit = dyn_cast<DagInit>(field)) {
    if (findUse(dagInit->getOperator(), deprecatedInit, known))
      return memoize(true);

    return memoize(llvm::any_of(dagInit->getArgs(), [&](const Init *arg) {
      return findUse(arg, deprecatedInit, known);
    }));
  }

  if (const ListInit *li = dyn_cast<ListInit>(field)) {
    return memoize(llvm::any_of(li->getValues(), [&](const Init *jt) {
      return findUse(jt, deprecatedInit, known);
    }));
  }

  // Purposefully don't use memoize here. There is no need to cache the result
  // for every kind of init (e.g. BitInit or StringInit), which will always
  // return false. Doing so would grow the DenseMap to include almost every Init
  // within the main file.
  return false;
}

// Returns if there is a use of `deprecatedInit` in `record`.
static bool findUse(Record &record, const Init *deprecatedInit,
                    llvm::DenseMap<const Init *, bool> &known) {
  return llvm::any_of(record.getValues(), [&](const RecordVal &val) {
    return findUse(val.getValue(), deprecatedInit, known);
  });
}

static void warnOfDeprecatedUses(const RecordKeeper &records) {
  // This performs a direct check for any def marked as deprecated and then
  // finds all uses of deprecated def. Deprecated defs are not expected to be
  // either numerous or long lived.
  bool deprecatedDefsFounds = false;
  for (auto &it : records.getDefs()) {
    const RecordVal *r = it.second->getValue("odsDeprecated");
    if (!r || !r->getValue())
      continue;

    llvm::DenseMap<const Init *, bool> hasUse;
    if (auto *si = dyn_cast<StringInit>(r->getValue())) {
      for (auto &jt : records.getDefs()) {
        // Skip anonymous defs.
        if (jt.second->isAnonymous())
          continue;

        if (findUse(*jt.second, it.second->getDefInit(), hasUse)) {
          PrintWarning(jt.second->getLoc(),
                       "Using deprecated def `" + it.first + "`");
          PrintNote(si->getAsUnquotedString());
          deprecatedDefsFounds = true;
        }
      }
    }
  }
  if (deprecatedDefsFounds &&
      actionOnDeprecatedValue == DeprecatedAction::Error)
    PrintFatalNote("Error'ing out due to deprecated defs");
}

// Generator to invoke.
static const mlir::GenInfo *generator;

// TableGenMain requires a function pointer so this function is passed in which
// simply wraps the call to the generator.
static bool mlirTableGenMain(raw_ostream &os, const RecordKeeper &records) {
  if (actionOnDeprecatedValue != DeprecatedAction::None)
    warnOfDeprecatedUses(records);

  if (!generator) {
    os << records;
    return false;
  }
  return generator->invoke(records, os);
}

int mlir::MlirTblgenMain(int argc, char **argv) {

  llvm::InitLLVM y(argc, argv);

  llvm::cl::opt<DeprecatedAction, true> actionOnDeprecated(
      "on-deprecated", llvm::cl::desc("Action to perform on deprecated def"),
      llvm::cl::values(
          clEnumValN(DeprecatedAction::None, "none", "No action"),
          clEnumValN(DeprecatedAction::Warn, "warn", "Warn on use"),
          clEnumValN(DeprecatedAction::Error, "error", "Error on use")),
      cl::location(actionOnDeprecatedValue), llvm::cl::init(Warn));

  llvm::cl::opt<const mlir::GenInfo *, true, mlir::GenNameParser> generator(
      "", llvm::cl::desc("Generator to run"), cl::location(::generator));

  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &mlirTableGenMain);
}
