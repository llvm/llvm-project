//===- TableGen.cpp - Top-Level TableGen implementation for LLVM ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the global defintions (mostly command line parameters)
// shared between llvm-tblgen and llvm-min-tblgen.
//
//===----------------------------------------------------------------------===//

#include "TableGen.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/SetTheory.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cassert>
#include <string>
#include <vector>

using namespace llvm;

namespace llvm {
cl::opt<bool> EmitLongStrLiterals(
    "long-string-literals",
    cl::desc("when emitting large string tables, prefer string literals over "
             "comma-separated char literals. This can be a readability and "
             "compile-time performance win, but upsets some compilers"),
    cl::Hidden, cl::init(true));
} // end namespace llvm

static cl::OptionCategory PrintEnumsCat("Options for -print-enums");
static cl::opt<std::string> Class("class",
                                  cl::desc("Print Enum list for this class"),
                                  cl::value_desc("class name"),
                                  cl::cat(PrintEnumsCat));

static void printRecords(const RecordKeeper &Records, raw_ostream &OS) {
  OS << Records; // No argument, dump all contents
}

static void printEnums(const RecordKeeper &Records, raw_ostream &OS) {
  for (const Record *Rec : Records.getAllDerivedDefinitions(Class))
    OS << Rec->getName() << ", ";
  OS << "\n";
}

static void printSets(const RecordKeeper &Records, raw_ostream &OS) {
  SetTheory Sets;
  Sets.addFieldExpander("Set", "Elements");
  for (const Record *Rec : Records.getAllDerivedDefinitions("Set")) {
    OS << Rec->getName() << " = [";
    const std::vector<const Record *> *Elts = Sets.expand(Rec);
    assert(Elts && "Couldn't expand Set instance");
    for (const Record *Elt : *Elts)
      OS << ' ' << Elt->getName();
    OS << " ]\n";
  }
}

static TableGen::Emitter::Opt X[] = {
    {"print-records", printRecords, "Print all records to stdout (default)",
     true},
    {"print-detailed-records", EmitDetailedRecords,
     "Print full details of all records to stdout"},
    {"null-backend", [](const RecordKeeper &Records, raw_ostream &OS) {},
     "Do nothing after parsing (useful for timing)"},
    {"dump-json", EmitJSON, "Dump all records as machine-readable JSON"},
    {"print-enums", printEnums, "Print enum values for a class"},
    {"print-sets", printSets, "Print expanded sets for testing DAG exprs"},
};

int tblgen_main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0]);
}

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(address_sanitizer) ||                                        \
    (defined(__SANITIZE_ADDRESS__) && defined(__GNUC__)) ||                    \
    __has_feature(leak_sanitizer)

#include <sanitizer/lsan_interface.h>
// Disable LeakSanitizer for this binary as it has too many leaks that are not
// very interesting to fix. See compiler-rt/include/sanitizer/lsan_interface.h .
LLVM_ATTRIBUTE_USED int __lsan_is_turned_off() { return 1; }

#endif
