//===- offload-tblgen/EntryPointGen.cpp - Tablegen backend for Offload ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Tablegen backend that produces the actual entry points for the
// Offload API. It serves as a place to integrate functionality like tracing
// and validation before dispatching to the actual implementations.
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"

#include "GenCommon.hpp"
#include "RecordTypes.hpp"

using namespace llvm;
using namespace offload::tblgen;

static void EmitValidationFunc(const FunctionRec &F, raw_ostream &OS) {
  OS << CommentsHeader;
  // Emit preamble
  OS << formatv("{0}_impl_result_t {1}_val(\n  ", PrefixLower, F.getName());
  // Emit arguments
  std::string ParamNameList = "";
  for (auto &Param : F.getParams()) {
    OS << Param.getType() << " " << Param.getName();
    if (Param != F.getParams().back()) {
      OS << ", ";
    }
    ParamNameList += Param.getName().str() + ", ";
  }
  OS << ") {\n";

  OS << TAB_1 "if (true /*enableParameterValidation*/) {\n";
  // Emit validation checks
  for (const auto &Return : F.getReturns()) {
    for (auto &Condition : Return.getConditions()) {
      if (Condition.starts_with("`") && Condition.ends_with("`")) {
        auto ConditionString = Condition.substr(1, Condition.size() - 2);
        OS << formatv(TAB_2 "if ({0}) {{\n", ConditionString);
        OS << formatv(TAB_3 "return {0};\n", Return.getValue());
        OS << TAB_2 "}\n\n";
      }
    }
  }
  OS << TAB_1 "}\n\n";

  // Perform actual function call to the implementation
  ParamNameList = ParamNameList.substr(0, ParamNameList.size() - 2);
  OS << formatv(TAB_1 "return {0}_impl({1});\n\n", F.getName(), ParamNameList);
  OS << "}\n";
}

static void EmitEntryPointFunc(const FunctionRec &F, raw_ostream &OS) {
  // Emit preamble
  OS << formatv("{1}_APIEXPORT {0}_result_t {1}_APICALL {2}(\n  ", PrefixLower,
                PrefixUpper, F.getName());
  // Emit arguments
  std::string ParamNameList = "";
  for (auto &Param : F.getParams()) {
    OS << Param.getType() << " " << Param.getName();
    if (Param != F.getParams().back()) {
      OS << ", ";
    }
    ParamNameList += Param.getName().str() + ", ";
  }
  OS << ") {\n";

  // Emit pre-call prints
  OS << TAB_1 "if (offloadConfig().TracingEnabled) {\n";
  OS << formatv(TAB_2 "std::cout << \"---> {0}\";\n", F.getName());
  OS << TAB_1 "}\n\n";

  // Perform actual function call to the validation wrapper
  ParamNameList = ParamNameList.substr(0, ParamNameList.size() - 2);
  OS << formatv(TAB_1 "{0}_result_t Result = {1}_val({2});\n\n", PrefixLower,
                F.getName(), ParamNameList);

  // Emit post-call prints
  OS << TAB_1 "if (offloadConfig().TracingEnabled) {\n";
  if (F.getParams().size() > 0) {
    OS << formatv(TAB_2 "{0} Params = {{", F.getParamStructName());
    for (const auto &Param : F.getParams()) {
      OS << "&" << Param.getName();
      if (Param != F.getParams().back()) {
        OS << ", ";
      }
    }
    OS << formatv("};\n");
    OS << TAB_2 "std::cout << \"(\" << &Params << \")\";\n";
  } else {
    OS << TAB_2 "std::cout << \"()\";\n";
  }
  OS << TAB_2 "std::cout << \"-> \" << Result << \"\\n\";\n";
  OS << TAB_2 "if (Result && Result->Details) {\n";
  OS << TAB_3 "std::cout << \"     *Error Details* \" << Result->Details "
              "<< \" \\n\";\n";
  OS << TAB_2 "}\n";
  OS << TAB_1 "}\n";

  OS << TAB_1 "return Result;\n";
  OS << "}\n";
}

static void EmitCodeLocWrapper(const FunctionRec &F, raw_ostream &OS) {
  // Emit preamble
  OS << formatv("{0}_result_t {1}WithCodeLoc(\n  ", PrefixLower, F.getName());
  // Emit arguments
  std::string ParamNameList = "";
  for (auto &Param : F.getParams()) {
    OS << Param.getType() << " " << Param.getName() << ", ";
    ParamNameList += Param.getName().str();
    if (Param != F.getParams().back()) {
      ParamNameList += ", ";
    }
  }
  OS << "ol_code_location_t *CodeLocation";
  OS << ") {\n";
  OS << TAB_1 "currentCodeLocation() = CodeLocation;\n";
  OS << formatv(TAB_1 "{0}_result_t Result = {1}({2});\n\n", PrefixLower,
                F.getName(), ParamNameList);
  OS << TAB_1 "currentCodeLocation() = nullptr;\n";
  OS << TAB_1 "return Result;\n";
  OS << "}\n";
}

void EmitOffloadEntryPoints(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader;
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    EmitValidationFunc(FunctionRec{R}, OS);
    EmitEntryPointFunc(FunctionRec{R}, OS);
    EmitCodeLocWrapper(FunctionRec{R}, OS);
  }
}
