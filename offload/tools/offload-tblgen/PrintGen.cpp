//===- offload-tblgen/APIGen.cpp - Tablegen backend for Offload printing --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Tablegen backend that produces print functions for the Offload API
// entry point functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"

#include "GenCommon.hpp"
#include "RecordTypes.hpp"

using namespace llvm;
using namespace offload::tblgen;

constexpr auto PrintEnumHeader =
    R"(///////////////////////////////////////////////////////////////////////////////
/// @brief Print operator for the {0} type
/// @returns std::ostream &
)";

constexpr auto PrintTaggedEnumHeader =
    R"(///////////////////////////////////////////////////////////////////////////////
/// @brief Print type-tagged {0} enum value
/// @returns std::ostream &
)";

static void ProcessEnum(const EnumRec &Enum, raw_ostream &OS) {
  OS << formatv(PrintEnumHeader, Enum.getName());
  OS << formatv(
      "inline std::ostream &operator<<(std::ostream &os, enum {0} value) "
      "{{\n" TAB_1 "switch (value) {{\n",
      Enum.getName());

  for (const auto &Val : Enum.getValues()) {
    auto Name = Enum.getEnumValNamePrefix() + "_" + Val.getName();
    OS << formatv(TAB_1 "case {0}:\n", Name);
    OS << formatv(TAB_2 "os << \"{0}\";\n", Name);
    OS << formatv(TAB_2 "break;\n");
  }

  OS << TAB_1 "default:\n" TAB_2 "os << \"unknown enumerator\";\n" TAB_2
              "break;\n" TAB_1 "}\n" TAB_1 "return os;\n}\n\n";

  if (!Enum.isTyped()) {
    return;
  }

  OS << formatv(PrintTaggedEnumHeader, Enum.getName());

  OS << formatv(R"""(template <>
inline void printTagged(std::ostream &os, const void *ptr, {0} value, size_t size) {{
  if (ptr == NULL) {{
    printPtr(os, ptr);
    return;
  }

  switch (value) {{
)""",
                Enum.getName());

  for (const auto &Val : Enum.getValues()) {
    auto Name = Enum.getEnumValNamePrefix() + "_" + Val.getName();
    auto Type = Val.getTaggedType();
    OS << formatv(TAB_1 "case {0}: {{\n", Name);
    // Special case for strings
    if (Type == "char[]") {
      OS << formatv(TAB_2 "printPtr(os, (const char*) ptr);\n");
    } else {
      OS << formatv(TAB_2 "const {0} * const tptr = (const {0} * const)ptr;\n",
                    Type);
      // TODO: Handle other cases here
      OS << TAB_2 "os << (const void *)tptr << \" (\";\n";
      if (Type.ends_with("*")) {
        OS << TAB_2 "os << printPtr(os, tptr);\n";
      } else {
        OS << TAB_2 "os << *tptr;\n";
      }
      OS << TAB_2 "os << \")\";\n";
    }
    OS << formatv(TAB_2 "break;\n" TAB_1 "}\n");
  }

  OS << TAB_1 "default:\n" TAB_2 "os << \"unknown enumerator\";\n" TAB_2
              "break;\n" TAB_1 "}\n";

  OS << "}\n";
}

static void EmitResultPrint(raw_ostream &OS) {
  OS << R""(
inline std::ostream &operator<<(std::ostream &os,
                                const ol_error_struct_t *Err) {
  if (Err == nullptr) {
    os << "OL_SUCCESS";
  } else {
    os << Err->Code;
  }
  return os;
}
)"";
}

static void EmitFunctionParamStructPrint(const FunctionRec &Func,
                                         raw_ostream &OS) {
  if (Func.getParams().size() == 0) {
    return;
  }

  OS << formatv(R"(
inline std::ostream &operator<<(std::ostream &os, const struct {0} *params) {{
)",
                Func.getParamStructName());

  for (const auto &Param : Func.getParams()) {
    OS << formatv(TAB_1 "os << \".{0} = \";\n", Param.getName());
    if (auto Range = Param.getRange()) {
      OS << formatv(TAB_1 "os << \"{{\";\n");
      OS << formatv(TAB_1 "for (size_t i = {0}; i < *params->p{1}; i++) {{\n",
                    Range->first, Range->second);
      OS << TAB_2 "if (i > 0) {\n";
      OS << TAB_3 " os << \", \";\n";
      OS << TAB_2 "}\n";
      OS << formatv(TAB_2 "printPtr(os, (*params->p{0})[i]);\n",
                    Param.getName());
      OS << formatv(TAB_1 "}\n");
      OS << formatv(TAB_1 "os << \"}\";\n");
    } else if (auto TypeInfo = Param.getTypeInfo()) {
      OS << formatv(
          TAB_1
          "printTagged(os, *params->p{0}, *params->p{1}, *params->p{2});\n",
          Param.getName(), TypeInfo->first, TypeInfo->second);
    } else if (Param.isPointerType() || Param.isHandleType()) {
      OS << formatv(TAB_1 "printPtr(os, *params->p{0});\n", Param.getName());
    } else {
      OS << formatv(TAB_1 "os << *params->p{0};\n", Param.getName());
    }
    if (Param != Func.getParams().back()) {
      OS << TAB_1 "os << \", \";\n";
    }
  }

  OS << TAB_1 "return os;\n}\n";
}

void EmitOffloadPrintHeader(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader;
  OS << R"""(
// Auto-generated file, do not manually edit.

#pragma once

#include <OffloadAPI.h>
#include <ostream>


template <typename T> inline ol_result_t printPtr(std::ostream &os, const T *ptr);
template <typename T> inline void printTagged(std::ostream &os, const void *ptr, T value, size_t size);
)""";

  // ==========
  OS << "template <typename T> struct is_handle : std::false_type {};\n";
  for (auto *R : Records.getAllDerivedDefinitions("Handle")) {
    HandleRec H{R};
    OS << formatv("template <> struct is_handle<{0}> : std::true_type {{};\n",
                  H.getName());
  }
  OS << "template <typename T> inline constexpr bool is_handle_v = "
        "is_handle<T>::value;\n";
  // =========

  // Forward declare the operator<< overloads so their implementations can
  // use each other.
  OS << "\n";
  for (auto *R : Records.getAllDerivedDefinitions("Enum")) {
    OS << formatv(
        "inline std::ostream &operator<<(std::ostream &os, enum {0} value);\n",
        EnumRec{R}.getName());
  }
  OS << "\n";

  // Create definitions
  for (auto *R : Records.getAllDerivedDefinitions("Enum")) {
    EnumRec E{R};
    ProcessEnum(E, OS);
  }
  EmitResultPrint(OS);

  // Emit print functions for the function param structs
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    EmitFunctionParamStructPrint(FunctionRec{R}, OS);
  }

  OS << R"""(
///////////////////////////////////////////////////////////////////////////////
// @brief Print pointer value
template <typename T> inline ol_result_t printPtr(std::ostream &os, const T *ptr) {
    if (ptr == nullptr) {
        os << "nullptr";
    } else if constexpr (std::is_pointer_v<T>) {
        os << (const void *)(ptr) << " (";
        printPtr(os, *ptr);
        os << ")";
    } else if constexpr (std::is_void_v<T> || is_handle_v<T *>) {
        os << (const void *)ptr;
    } else if constexpr (std::is_same_v<std::remove_cv_t< T >, char>) {
        os << (const void *)(ptr) << " (";
        os << ptr;
        os << ")";
    } else {
        os << (const void *)(ptr) << " (";
        os << *ptr;
        os << ")";
    }

    return OL_SUCCESS;
}
  )""";
}
