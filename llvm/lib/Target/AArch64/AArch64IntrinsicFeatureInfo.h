//===- AArch64IntrinsicFeatureInfo.h - Intrinsic feature table -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64INTRINSICFEATUREINFO_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64INTRINSICFEATUREINFO_H

namespace llvm {

class GuardedIntrinsicInfo;

/// Return the AArch64 intrinsic feature table shared by backend passes.
const GuardedIntrinsicInfo &getAArch64GuardedIntrinsicInfo();

} // namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_AARCH64INTRINSICFEATUREINFO_H
