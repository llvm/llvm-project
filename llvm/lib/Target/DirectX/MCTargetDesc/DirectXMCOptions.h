//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_MCTARGETDESC_DIRECTXMCOPTIONS_H
#define LLVM_LIB_TARGET_DIRECTX_MCTARGETDESC_DIRECTXMCOPTIONS_H

#include "llvm/MC/MCTargetOptions.h"

#define OPTIONS_STRUCT_DECL
#include "DirectXMCOptions.inc"

namespace llvm::DirectX_MC {
inline const DirectXMCOptions &getCLOpts(const MCTargetOptions &Options) {
  return cl::LibraryOptionsContext::get<DirectXMCOptions>(
      Options.LibraryOptions);
}
} // namespace llvm::DirectX_MC

#endif // LLVM_LIB_TARGET_DIRECTX_MCTARGETDESC_DIRECTXMCOPTIONS_H
