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
enum class OMPScheduleType {
  // For typed comparisons, not a valid schedule
  None = 0,

  // Schedule algorithms
  BaseStaticChunked = 1,
  BaseStatic = 2,
  BaseDynamicChunked = 3,
  BaseGuidedChunked = 4,
  BaseRuntime = 5,
  BaseAuto = 6,
  BaseTrapezoidal = 7,
  BaseGreedy = 8,
  BaseBalanced = 9,
  BaseGuidedIterativeChunked = 10,
  BaseGuidedAnalyticalChunked = 11,
  BaseSteal = 12,

  // with chunk adjustment (e.g., simd)
  BaseStaticBalancedChunked = 13,
  BaseGuidedSimd = 14,
  BaseRuntimeSimd = 15,

  // static schedules algorithims for distribute
  BaseDistributeChunked = 27,
  BaseDistribute = 28,

  // Modifier flags to be combined with schedule algorithms
  ModifierUnordered = (1 << 5),
  ModifierOrdered = (1 << 6),
  ModifierNomerge = (1 << 7),
  ModifierMonotonic = (1 << 29),
  ModifierNonmonotonic = (1 << 30),

  // Masks combining multiple flags
  OrderingMask = ModifierUnordered | ModifierOrdered | ModifierNomerge,
  MonotonicityMask = ModifierMonotonic | ModifierNonmonotonic,
  ModifierMask = OrderingMask | MonotonicityMask,

  // valid schedule type values, without monotonicity flags
  UnorderedStaticChunked = BaseStaticChunked | ModifierUnordered,   // 33
  UnorderedStatic = BaseStatic | ModifierUnordered,                 // 34
  UnorderedDynamicChunked = BaseDynamicChunked | ModifierUnordered, // 35
  UnorderedGuidedChunked = BaseGuidedChunked | ModifierUnordered,   // 36
  UnorderedRuntime = BaseRuntime | ModifierUnordered,               // 37
  UnorderedAuto = BaseAuto | ModifierUnordered,                     // 38
  UnorderedTrapezoidal = BaseTrapezoidal | ModifierUnordered,       // 39
  UnorderedGreedy = BaseGreedy | ModifierUnordered,                 // 40
  UnorderedBalanced = BaseBalanced | ModifierUnordered,             // 41
  UnorderedGuidedIterativeChunked =
      BaseGuidedIterativeChunked | ModifierUnordered, // 42
  UnorderedGuidedAnalyticalChunked =
      BaseGuidedAnalyticalChunked | ModifierUnordered, // 43
  UnorderedSteal = BaseSteal | ModifierUnordered,      // 44

  UnorderedStaticBalancedChunked =
      BaseStaticBalancedChunked | ModifierUnordered,          // 45
  UnorderedGuidedSimd = BaseGuidedSimd | ModifierUnordered,   // 46
  UnorderedRuntimeSimd = BaseRuntimeSimd | ModifierUnordered, // 47

  OrderedStaticChunked = BaseStaticChunked | ModifierOrdered,   // 65
  OrderedStatic = BaseStatic | ModifierOrdered,                 // 66
  OrderedDynamicChunked = BaseDynamicChunked | ModifierOrdered, // 67
  OrderedGuidedChunked = BaseGuidedChunked | ModifierOrdered,   // 68
  OrderedRuntime = BaseRuntime | ModifierOrdered,               // 69
  OrderedAuto = BaseAuto | ModifierOrdered,                     // 70
  OrderdTrapezoidal = BaseTrapezoidal | ModifierOrdered,        // 71

  OrderedDistributeChunked = BaseDistributeChunked | ModifierOrdered, // 91
  OrderedDistribute = BaseDistribute | ModifierOrdered,               // 92

  NomergeUnorderedStaticChunked =
      BaseStaticChunked | ModifierUnordered | ModifierNomerge, // 161
  NomergeUnorderedStatic =
      BaseStatic | ModifierUnordered | ModifierNomerge, // 162
  NomergeUnorderedDynamicChunked =
      BaseDynamicChunked | ModifierUnordered | ModifierNomerge, // 163
  NomergeUnorderedGuidedChunked =
      BaseGuidedChunked | ModifierUnordered | ModifierNomerge, // 164
  NomergeUnorderedRuntime =
      BaseRuntime | ModifierUnordered | ModifierNomerge,                 // 165
  NomergeUnorderedAuto = BaseAuto | ModifierUnordered | ModifierNomerge, // 166
  NomergeUnorderedTrapezoidal =
      BaseTrapezoidal | ModifierUnordered | ModifierNomerge, // 167
  NomergeUnorderedGreedy =
      BaseGreedy | ModifierUnordered | ModifierNomerge, // 168
  NomergeUnorderedBalanced =
      BaseBalanced | ModifierUnordered | ModifierNomerge, // 169
  NomergeUnorderedGuidedIterativeChunked =
      BaseGuidedIterativeChunked | ModifierUnordered | ModifierNomerge, // 170
  NomergeUnorderedGuidedAnalyticalChunked =
      BaseGuidedAnalyticalChunked | ModifierUnordered | ModifierNomerge, // 171
  NomergeUnorderedSteal =
      BaseSteal | ModifierUnordered | ModifierNomerge, // 172

  NomergeOrderedStaticChunked =
      BaseStaticChunked | ModifierOrdered | ModifierNomerge,             // 193
  NomergeOrderedStatic = BaseStatic | ModifierOrdered | ModifierNomerge, // 194
  NomergeOrderedDynamicChunked =
      BaseDynamicChunked | ModifierOrdered | ModifierNomerge, // 195
  NomergeOrderedGuidedChunked =
      BaseGuidedChunked | ModifierOrdered | ModifierNomerge, // 196
  NomergeOrderedRuntime =
      BaseRuntime | ModifierOrdered | ModifierNomerge,               // 197
  NomergeOrderedAuto = BaseAuto | ModifierOrdered | ModifierNomerge, // 198
  NomergeOrderedTrapezoidal =
      BaseTrapezoidal | ModifierOrdered | ModifierNomerge, // 199

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

/// \note This needs to be kept in sync with interop.h enum kmp_interop_type_t.:
enum class OMPInteropType { Unknown, Target, TargetSync };

/// Atomic compare operations. Currently OpenMP only supports ==, >, and <.
enum class OMPAtomicCompareOp : unsigned { EQ, MIN, MAX };

} // end namespace omp

} // end namespace llvm

#endif // LLVM_FRONTEND_OPENMP_OMPCONSTANTS_H
