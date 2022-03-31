//===- mlir-tblgen.cpp - Top-Level TableGen implementation for MLIR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for MLIR's TableGen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

enum DeprecatedAction { None, Warn, Error };
llvm::cl::opt<DeprecatedAction> actionOnDeprecated(
    "on-deprecated", llvm::cl::init(Warn),
    llvm::cl::desc("Action to perform on deprecated def"),
    llvm::cl::values(clEnumValN(DeprecatedAction::None, "none", "No action"),
                     clEnumValN(DeprecatedAction::Warn, "warn", "Warn on use"),
                     clEnumValN(DeprecatedAction::Error, "error",
                                "Error on use")));

static llvm::ManagedStatic<std::vector<GenInfo>> generatorRegistry;

mlir::GenRegistration::GenRegistration(StringRef arg, StringRef description,
                                       const GenFunction &function) {
  generatorRegistry->emplace_back(arg, description, function);
}

GenNameParser::GenNameParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const GenInfo *>(opt) {
  for (const auto &kv : *generatorRegistry) {
    addLiteralOption(kv.getGenArgument(), &kv, kv.getGenDescription());
  }
}

void GenNameParser::printOptionInfo(const llvm::cl::Option &o,
                                    size_t globalWidth) const {
  GenNameParser *tp = const_cast<GenNameParser *>(this);
  llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                       [](const GenNameParser::OptionInfo *vT1,
                          const GenNameParser::OptionInfo *vT2) {
                         return vT1->Name.compare(vT2->Name);
                       });
  using llvm::cl::parser;
  parser<const GenInfo *>::printOptionInfo(o, globalWidth);
}

// Generator that prints records.
GenRegistration printRecords("print-records", "Print all records to stdout",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               os << records;
                               return false;
                             });

// Generator to invoke.
const mlir::GenInfo *generator;

// Returns if there is a use of `init` in `record`.
bool findUse(Record &record, Init *init,
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

void warnOfDeprecatedUses(RecordKeeper &records) {
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
  if (deprecatedDefsFounds && actionOnDeprecated == DeprecatedAction::Error)
    PrintFatalNote("Error'ing out due to deprecated defs");
}

// TableGenMain requires a function pointer so this function is passed in which
// simply wraps the call to the generator.
static bool mlirTableGenMain(raw_ostream &os, RecordKeeper &records) {
  if (actionOnDeprecated != DeprecatedAction::None)
    warnOfDeprecatedUses(records);

  if (!generator) {
    os << records;
    return false;
  }
  return generator->invoke(records, os);
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::opt<const mlir::GenInfo *, false, mlir::GenNameParser> generator(
      "", llvm::cl::desc("Generator to run"));
  cl::ParseCommandLineOptions(argc, argv);
  ::generator = generator.getValue();

  return TableGenMain(argv[0], &mlirTableGenMain);
}
