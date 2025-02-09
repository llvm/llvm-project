//===- offload-tblgen/APIGen.cpp - Tablegen backend for Offload functions -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Tablegen backend that handles generation of various small files
// pertaining to the API functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"

#include "GenCommon.hpp"
#include "RecordTypes.hpp"

using namespace llvm;
using namespace offload::tblgen;

// Emit a list of just the API function names
void EmitOffloadFuncNames(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader;
  OS << R"(
#ifndef OFFLOAD_FUNC
#error Please define the macro OFFLOAD_FUNC(Function)
#endif

)";
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    FunctionRec FR{R};
    OS << formatv("OFFLOAD_FUNC({0})", FR.getName()) << "\n";
  }
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    FunctionRec FR{R};
    OS << formatv("OFFLOAD_FUNC({0}WithCodeLoc)", FR.getName()) << "\n";
  }

  OS << "\n#undef OFFLOAD_FUNC\n";
}

void EmitOffloadExports(const RecordKeeper &Records, raw_ostream &OS) {
  OS << "VERS1.0 {\n";
  OS << TAB_1 "global:\n";

  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    OS << formatv(TAB_2 "{0};\n", FunctionRec(R).getName());
  }
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    OS << formatv(TAB_2 "{0}WithCodeLoc;\n", FunctionRec(R).getName());
  }
  OS << TAB_1 "local:\n";
  OS << TAB_2 "*;\n";
  OS << "};\n";
}

// Emit declarations for every implementation function
void EmitOffloadImplFuncDecls(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader;
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    FunctionRec F{R};
    OS << formatv("{0}_impl_result_t {1}_impl(", PrefixLower, F.getName());
    auto Params = F.getParams();
    for (auto &Param : Params) {
      OS << Param.getType() << " " << Param.getName();
      if (Param != Params.back()) {
        OS << ", ";
      }
    }
    OS << ");\n\n";
  }
}
