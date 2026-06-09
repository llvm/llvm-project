//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_TARGETPARSER_H
#define LLVM_TARGETPARSER_TARGETPARSER_H

#include "SubtargetFeature.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {

struct BasicSubtargetFeatureKV {
  const char *Key;         ///< K-V key string
  unsigned Value;          ///< K-V integer value
  FeatureBitsetIndex Implies; ///< Index into transitive feature mask table
};
static_assert(sizeof(void *) != 8 || sizeof(BasicSubtargetFeatureKV) == 16);
static_assert(sizeof(void *) != 4 || sizeof(BasicSubtargetFeatureKV) == 12);

/// Used to provide key value pairs for feature and CPU bit flags.
struct BasicSubtargetSubTypeKV {
  const char *Key;         ///< K-V key string
  FeatureBitsetIndex Implies; ///< Index into transitive feature mask table

  /// Compare routine for std::lower_bound
  bool operator<(StringRef S) const { return StringRef(Key) < S; }

  /// Compare routine for std::is_sorted.
  bool operator<(const BasicSubtargetSubTypeKV &Other) const {
    return StringRef(Key) < StringRef(Other.Key);
  }
};
static_assert(sizeof(void *) != 8 || sizeof(BasicSubtargetSubTypeKV) == 16);
static_assert(sizeof(void *) != 4 || sizeof(BasicSubtargetSubTypeKV) == 8);

LLVM_ABI std::optional<llvm::StringMap<bool>>
getCPUDefaultTargetFeatures(StringRef CPU,
                            ArrayRef<BasicSubtargetSubTypeKV> ProcDesc,
                            ArrayRef<BasicSubtargetFeatureKV> ProcFeatures,
                            ArrayRef<FeatureBitArray> FeatureMasks);
} // namespace llvm

#endif
