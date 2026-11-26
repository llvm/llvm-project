//===-- SparcTargetParser.cpp - Parser for SPARC target features ----------===//
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

#include "llvm/TargetParser/SparcTargetParser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TargetParser/TargetParser.h"

#define GET_SUBTARGETFEATURES_ENUM
#define GET_SUBTARGETFEATURES_KV
#include "llvm/TargetParser/SparcGenTargetFeatures.inc"

namespace llvm {
namespace Sparc {

bool isValidCPU(StringRef CPU) {
  ArrayRef<BasicSubtargetSubTypeKV> A = BasicSparcSubTypeKV;
  const BasicSubtargetSubTypeKV *P = llvm::lower_bound(A, CPU);
  return P != A.end() && StringRef(P->Key) == CPU;
}

void fillValidCPUList(SmallVectorImpl<StringRef> &Values) {
  for (const BasicSubtargetSubTypeKV &P : BasicSparcSubTypeKV)
    Values.emplace_back(P.Key);
}

StringRef getNormalizedSparcTargetCPU(const Triple &T, StringRef CPUName) {
  if (!CPUName.empty())
    return CPUName;
  return T.isSPARC64() ? "v9" : "v8";
}

std::optional<StringMap<bool>> getSparcDefaultTargetFeatures(StringRef CPU) {
  return getCPUDefaultTargetFeatures(CPU, BasicSparcSubTypeKV,
                                     BasicSparcFeatureKV);
}

bool isValidFeatureName(StringRef Name) {
  ArrayRef<BasicSubtargetFeatureKV> A = BasicSparcFeatureKV;
  const BasicSubtargetFeatureKV *F = llvm::lower_bound(A, Name);
  return F != A.end() && StringRef(F->Key) == Name;
}

} // namespace Sparc
} // namespace llvm
