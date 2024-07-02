//===- offload-tblgen/offload-tblgen.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Tablegen tool that produces source files for the Offload project.
// See offload/API/README.md for more information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

#include "Generators.hpp"

namespace llvm {
namespace offload {
namespace tblgen {

enum ActionType { PrintRecords, DumpJSON, GenAPI, GenValidation };

namespace {
cl::opt<ActionType> Action(
    cl::desc("Action to perform:"),
    cl::values(
        clEnumValN(PrintRecords, "print-records",
                   "Print all records to stdout (default)"),
        clEnumValN(DumpJSON, "dump-json",
                   "Dump all records as machine-readable JSON"),
        clEnumValN(GenAPI, "gen-api", "Generate Offload API header contents"),
        clEnumValN(GenValidation, "gen-validation",
                   "Generate Offload entry point validation functions")));
}

static bool OffloadTableGenMain(raw_ostream &OS, RecordKeeper &Records) {
  switch (Action) {
  case PrintRecords:
    OS << Records;
    break;
  case DumpJSON:
    EmitJSON(Records, OS);
    break;
  case GenAPI:
    EmitOffloadAPI(Records, OS);
    break;
  case GenValidation:
    EmitOffloadValidation(Records, OS);
    break;
  default:
    break;
  }

  return false;
}

int OffloadTblgenMain(int argc, char **argv) {
  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &OffloadTableGenMain);
  ;
}
} // namespace tblgen
} // namespace offload
} // namespace llvm

using namespace llvm;
using namespace offload::tblgen;

int main(int argc, char **argv) { return OffloadTblgenMain(argc, argv); }
