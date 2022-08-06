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

// Returns if there is a use of `init` in `record`.
static bool findUse(Record &record, Init *init,
                    llvm::DenseMap<Record *, bool> &known) {
  auto it = known.find(&record);
  if (it != known.end())
    return it->second;

  auto memoize = [&](bool val) {
    known[&record] = val;
    return val;
  };

  for (const RecordVal &val : record.getValues()) {
    Init *valInit = val.getValue();
    if (valInit == init)
      return true;
    if (auto *di = dyn_cast<DefInit>(valInit)) {
      if (findUse(*di->getDef(), init, known))
        return memoize(true);
    } else if (auto *di = dyn_cast<DagInit>(valInit)) {
      for (Init *arg : di->getArgs())
        if (auto *di = dyn_cast<DefInit>(arg))
          if (findUse(*di->getDef(), init, known))
            return memoize(true);
    } else if (ListInit *li = dyn_cast<ListInit>(valInit)) {
      for (Init *jt : li->getValues())
        if (jt == init)
          return memoize(true);
    }
  }
  return memoize(false);
}

static void warnOfDeprecatedUses(RecordKeeper &records) {
  // This performs a direct check for any def marked as deprecated and then
  // finds all uses of deprecated def. Deprecated defs are not expected to be
  // either numerous or long lived.
  bool deprecatedDefsFounds = false;
  for (auto &it : records.getDefs()) {
    const RecordVal *r = it.second->getValue("odsDeprecated");
    if (!r || !r->getValue())
      continue;

    llvm::DenseMap<Record *, bool> hasUse;
    if (auto *si = dyn_cast<StringInit>(r->getValue())) {
      for (auto &jt : records.getDefs()) {
        // Skip anonymous defs.
        if (jt.second->isAnonymous())
          continue;
        // Skip all outside main file to avoid flagging redundantly.
        unsigned buf =
            SrcMgr.FindBufferContainingLoc(jt.second->getLoc().front());
        if (buf != SrcMgr.getMainFileID())
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
static bool mlirTableGenMain(raw_ostream &os, RecordKeeper &records) {
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
