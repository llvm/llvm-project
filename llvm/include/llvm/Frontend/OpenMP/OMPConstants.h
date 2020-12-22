//===- OMPConstants.h - OpenMP related constants and helpers ------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines constans and helpers used when dealing with OpenMP.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_CONSTANTS_H
#define LLVM_OPENMP_CONSTANTS_H

#include "llvm/ADT/BitmaskEnum.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h.inc"

namespace llvm {
class Type;
class Module;
class ArrayType;
class StructType;
class PointerType;
class StringRef;
class FunctionType;

namespace omp {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// IDs for all Internal Control Variables (ICVs).
enum class InternalControlVar {
#define ICV_DATA_ENV(Enum, ...) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

#define ICV_DATA_ENV(Enum, ...)                                                \
  constexpr auto Enum = omp::InternalControlVar::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

enum class ICVInitValue {
#define ICV_INIT_VALUE(Enum, Name) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

#define ICV_INIT_VALUE(Enum, Name)                                             \
  constexpr auto Enum = omp::ICVInitValue::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// IDs for all omp runtime library (RTL) functions.
enum class RuntimeFunction {
#define OMP_RTL(Enum, ...) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

#define OMP_RTL(Enum, ...) constexpr auto Enum = omp::RuntimeFunction::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// IDs for the different default kinds.
enum class DefaultKind {
#define OMP_DEFAULT_KIND(Enum, Str) Enum,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

#define OMP_DEFAULT_KIND(Enum, ...)                                            \
  constexpr auto Enum = omp::DefaultKind::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// IDs for all omp runtime library ident_t flag encodings (see
/// their defintion in openmp/runtime/src/kmp.h).
enum class IdentFlag {
#define OMP_IDENT_FLAG(Enum, Str, Value) Enum = Value,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  LLVM_MARK_AS_BITMASK_ENUM(0x7FFFFFFF)
};

#define OMP_IDENT_FLAG(Enum, ...) constexpr auto Enum = omp::IdentFlag::Enum;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// Helper to describe assume clauses.
struct AssumptionClauseMappingInfo {
  /// The identifier describing the (beginning of the) clause.
  llvm::StringLiteral Identifier;
  /// Flag to determine if the identifier is a full name or the start of a name.
  bool StartsWith;
  /// Flag to determine if a directive lists follows.
  bool HasDirectiveList;
  /// Flag to determine if an expression follows.
  bool HasExpression;
};

/// All known assume clauses.
static constexpr AssumptionClauseMappingInfo AssumptionClauseMappings[] = {
#define OMP_ASSUME_CLAUSE(Identifier, StartsWith, HasDirectiveList,            \
                          HasExpression)                                       \
  {Identifier, StartsWith, HasDirectiveList, HasExpression},
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

inline std::string getAllAssumeClauseOptions() {
  std::string S;
  for (const AssumptionClauseMappingInfo &ACMI : AssumptionClauseMappings)
    S += (S.empty() ? "'" : "', '") + ACMI.Identifier.str();
  return S + "'";
}

} // end namespace omp

} // end namespace llvm

#endif // LLVM_OPENMP_CONSTANTS_H
