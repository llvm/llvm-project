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
    OS << formatv("Error {0}_impl(", F.getName());
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

// Emit macro calls for each error enum
void EmitOffloadErrcodes(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader;
  OS << R"(
#ifndef OFFLOAD_ERRC
#error Please define the macro OFFLOAD_ERRCODE(Name, Desc, Value)
#endif

// Error codes are shared between PluginInterface and liboffload.
// To add new error codes, add them to offload/liboffload/API/Common.td.

)";

  auto ErrorCodeEnum = EnumRec{Records.getDef("ol_errc_t")};
  uint32_t EtorVal = 0;
  for (const auto &EnumVal : ErrorCodeEnum.getValues()) {
    OS << formatv(TAB_1 "OFFLOAD_ERRC({0}, \"{1}\", {2})\n", EnumVal.getName(),
                  EnumVal.getDesc(), EtorVal++);
  }
}

// Emit macro calls for each info
void EmitOffloadInfo(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader;
  OS << R"(
#ifndef OFFLOAD_DEVINFO
#error Please define the macro OFFLOAD_DEVINFO(Name, Desc, Value)
#endif

// Device info codes are shared between PluginInterface and liboffload.
// To add new error codes, add them to offload/liboffload/API/Device.td.

)";

  auto Enum = EnumRec{Records.getDef("ol_device_info_t")};
  // Bitfields start from 1, other enums from 0
  uint32_t EtorVal = Enum.isBitField();
  for (const auto &EnumVal : Enum.getValues()) {
    OS << formatv(TAB_1 "OFFLOAD_DEVINFO({0}, \"{1}\", {2})\n",
                  EnumVal.getName(), EnumVal.getDesc(), EtorVal);
    if (Enum.isBitField()) {
      EtorVal <<= 1u;
    } else {
      ++EtorVal;
    }
  }
}
