//===- offload-tblgen/APIGen.cpp - Tablegen backend for Offload validation ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Tablegen backend that produces validation functions for the Offload
// API entry point functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"

#include "GenCommon.hpp"
#include "RecordTypes.hpp"

using namespace llvm;
using namespace offload::tblgen;

static void EmitValidationFunc(const FunctionRec &F, raw_ostream &OS) {
  OS << CommentsHeader;
  OS << formatv("/// @brief Intercept function for {0}\n", F.getFullName());
  // Emit preamble
  OS << formatv("{0}_result_t {1}_APICALL val_{2}(\n", PrefixLower, PrefixUpper,
                F.getFullName());
  // Emit arguments
  std::string ParamNameList = "";
  for (auto &Param : F.getParams()) {
    OS << "  " << Param.getType() << " " << Param.getName();
    if (Param != F.getParams().back()) {
      OS << ", ";
    } else {
      OS << " ";
    }
    OS << MakeParamComment(Param) << "\n";
    ParamNameList += Param.getName().str() + ", ";
  }
  OS << ") {\n";

  OS << "  if (true /*enableParameterValidation*/) {\n";

  // Emit validation checks
  for (const auto &Return : F.getReturns()) {
    for (auto &Condition : Return.getConditions()) {
      if (Condition.starts_with("`") && Condition.ends_with("`")) {
        auto ConditionString = Condition.substr(1, Condition.size() - 2);
        OS << formatv("    if ({0}) {{\n", ConditionString);
        OS << formatv("      return {0};\n", Return.getValue());
        OS << "    }\n\n";
      }
    }
  }
  OS << "  }\n\n";

  auto LifetimeTodoComment =
      R"(  // TODO: Implement. `refCountContext` is some global object that tracks known
  // live handle objects, and logs related errors.
  // In UR this is implemented as an unordered_map of handles to structs
  // containing the reference count, amongst other details. In this case, a 
  // handle is invalid if it does not exist in the map.
  
)";
  bool EmittedTodo = false;

  // Emit handle lifetime checks
  for (auto &Param : F.getParams()) {
    if (Param.getType().ends_with("handle_t")) {
      // Only add this comment once per function to keep the code size down
      if (!EmittedTodo) {
        OS << LifetimeTodoComment;
        EmittedTodo = true;
      }
      OS << formatv("  if (true /* enableLifeTimeValidation && "
                    "!refCountContext.isReferenceValid({0}) */) {{\n",
                    Param.getName());
      OS << formatv("    // refCountContext.logInvalidReference({0});\n",
                    Param.getName());
      OS << "  }\n\n";
    }
  }

  // Perform actual function call
  ParamNameList = ParamNameList.substr(0, ParamNameList.size() - 2);
  OS << formatv("  {0}_result_t result = {1}({2});\n\n", PrefixLower,
                F.getFullName(), ParamNameList);

  // Handle reference counting for cases where the function modifies the ref
  // count of a handle
  // * `Create` - initialize a reference count
  // * `Retain` - increment a reference count
  // * `Release` - decerement a reference count
  if (F.modifiesRefCount()) {
    OS << formatv("  if ( /*context.enableLeakChecking &&*/ result == "
                  "{0}_RESULT_SUCCESS) {\n",
                  PrefixUpper);

    // The refcount context optionally takes a bool specifying whether the
    // handle being tracked is an adapter handle, as they are counted
    // differently.
    // TODO: This behavior is lifted from UR. Offload will likely be different.
    auto AdapterHandleArg = (F.getClass() == "Adapter") ? "true" : "false";

    if (F.getName() == "Create") {
      // We only expect one handle output for these types of functions, but loop
      // over all params just in case
      for (auto &Param : F.getParams()) {
        if (Param.isOut()) {
          OS << formatv("    // refCountContext.createRefCount(*{0});\n",
                        Param.getName());
        }
      }
      // Retain and release functions only have 1 parameter
    } else if (F.getName() == "Retain") {
      OS << formatv("    // refCountContext.incrementRefCount({0}, {1});\n",
                    F.getParams().at(0).getName(), AdapterHandleArg);
    } else {
      OS << formatv("    // refCountContext.decrementRefCount({0}, {1});\n",
                    F.getParams().at(0).getName(), AdapterHandleArg);
    }
    OS << "  }\n";
  }

  OS << "  return result;\n";
  OS << "}\n";
}

void EmitOffloadValidation(RecordKeeper &Records, raw_ostream &OS) {
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    EmitValidationFunc(FunctionRec{R}, OS);
  }
}
