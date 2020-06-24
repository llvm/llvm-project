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
#define ICV_DATA_ENV(Enum, Name, EnvVar, Init) Init,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

#define ICV_DATA_ENV(Enum, Name, EnvVar, Init)                                 \
  constexpr auto Init = omp::ICVInitValue::Init;
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

/// IDs for the different proc bind kinds.
enum class ProcBindKind {
#define OMP_PROC_BIND_KIND(Enum, Str, Value) Enum = Value,
#include "llvm/Frontend/OpenMP/OMPKinds.def"
};

#define OMP_PROC_BIND_KIND(Enum, ...)                                          \
  constexpr auto Enum = omp::ProcBindKind::Enum;
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

/// Parse \p Str and return the directive it matches or OMPD_unknown if none.
Directive getOpenMPDirectiveKind(StringRef Str);

/// Return a textual representation of the directive \p D.
StringRef getOpenMPDirectiveName(Directive D);

/// Parse \p Str and return the clause it matches or OMPC_unknown if none.
Clause getOpenMPClauseKind(StringRef Str);

/// Return a textual representation of the clause \p C.
StringRef getOpenMPClauseName(Clause C);

/// Return true if \p C is a valid clause for \p D in version \p Version.
bool isAllowedClauseForDirective(Directive D, Clause C, unsigned Version);

/// Forward declarations for LLVM-IR types (simple, function and structure) are
/// generated below. Their names are defined and used in OpenMP/OMPKinds.def.
/// Here we provide the forward declarations, the initializeTypes function will
/// provide the values.
///
///{
namespace types {

#define OMP_TYPE(VarName, InitValue) extern Type *VarName;
#define OMP_ARRAY_TYPE(VarName, ElemTy, ArraySize)                             \
  extern ArrayType *VarName##Ty;                                               \
  extern PointerType *VarName##PtrTy;
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  extern FunctionType *VarName;                                                \
  extern PointerType *VarName##Ptr;
#define OMP_STRUCT_TYPE(VarName, StrName, ...)                                 \
  extern StructType *VarName;                                                  \
  extern PointerType *VarName##Ptr;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

/// Helper to initialize all types defined in OpenMP/OMPKinds.def.
void initializeTypes(Module &M);

/// Helper to uninitialize all types defined in OpenMP/OMPKinds.def.
void uninitializeTypes();

} // namespace types
///}

} // end namespace omp

} // end namespace llvm

#endif // LLVM_OPENMP_CONSTANTS_H
