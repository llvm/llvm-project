//===-- SparcTargetParser - Parser for SPARC target features ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features
// for SPARC CPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_SPARCTARGETPARSER_H
#define LLVM_TARGETPARSER_SPARCTARGETPARSER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

namespace llvm {
namespace Sparc {

/// Return true if \p CPU is a processor known to the SPARC backend.
LLVM_ABI bool isValidCPU(StringRef CPU);

/// Fill \p Values with the names of the processors known to the SPARC
/// backend.
LLVM_ABI void fillValidCPUList(SmallVectorImpl<StringRef> &Values);

/// Return \p CPUName, or the default CPU for \p T if \p CPUName is empty.
LLVM_ABI StringRef getNormalizedSparcTargetCPU(const Triple &T,
                                               StringRef CPUName = "");

/// Return the features enabled by default for \p CPU, including the ones
/// they imply, or std::nullopt if \p CPU is not a known processor.
LLVM_ABI std::optional<StringMap<bool>>
getSparcDefaultTargetFeatures(StringRef CPU);

/// Return true if \p Name is a SPARC target feature.
LLVM_ABI bool isValidFeatureName(StringRef Name);

} // namespace Sparc
} // namespace llvm

#endif
