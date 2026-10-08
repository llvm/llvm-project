//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64MCOPTIONS_H
#define LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64MCOPTIONS_H

#include <optional>

namespace llvm::AArch64 {
enum AsmWriterVariantTy { Generic = 0, Apple = 1 };
} // namespace llvm::AArch64

#define OPTIONS_STRUCT_DECL
#include "AArch64MCOptions.inc"

#endif // LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64MCOPTIONS_H
