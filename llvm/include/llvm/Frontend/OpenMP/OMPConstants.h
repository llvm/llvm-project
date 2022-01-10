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

#ifndef LLVM_FRONTEND_OPENMP_OMPCONSTANTS_H
#define LLVM_FRONTEND_OPENMP_OMPCONSTANTS_H

#include "llvm/ADT/BitmaskEnum.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h.inc"

namespace llvm {
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

/// \note This needs to be kept in sync with kmp.h enum sched_type.
/// Todo: Update kmp.h to include this file, and remove the enums in kmp.h
///       To complete this, more enum values will need to be moved here.
enum class OMPScheduleType {
  StaticChunked = 33,
  Static = 34, // static unspecialized
  DistributeChunked = 91,
  Distribute = 92,
  DynamicChunked = 35,
  GuidedChunked = 36, // guided unspecialized
  Runtime = 37,
  Auto = 38, // auto

  StaticBalancedChunked = 45, // static with chunk adjustment (e.g., simd)
  GuidedSimd = 46,            // guided with chunk adjustment
  RuntimeSimd = 47,           // runtime with chunk adjustment

  ModifierMonotonic =
      (1 << 29), // Set if the monotonic schedule modifier was present
  ModifierNonmonotonic =
      (1 << 30), // Set if the nonmonotonic schedule modifier was present
  ModifierMask = ModifierMonotonic | ModifierNonmonotonic,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue */ ModifierMask)
};

enum OMPTgtExecModeFlags : int8_t {
  OMP_TGT_EXEC_MODE_GENERIC = 1 << 0,
  OMP_TGT_EXEC_MODE_SPMD = 1 << 1,
  OMP_TGT_EXEC_MODE_GENERIC_SPMD =
      OMP_TGT_EXEC_MODE_GENERIC | OMP_TGT_EXEC_MODE_SPMD,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue */ OMP_TGT_EXEC_MODE_GENERIC_SPMD)
};

enum class AddressSpace : unsigned {
  Generic = 0,
  Global = 1,
  Shared = 3,
  Constant = 4,
  Local = 5,
};

} // end namespace omp

} // end namespace llvm

#endif // LLVM_FRONTEND_OPENMP_OMPCONSTANTS_H
