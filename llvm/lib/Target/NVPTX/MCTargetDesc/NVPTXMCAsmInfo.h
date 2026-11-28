//===-- NVPTXMCAsmInfo.h - NVPTX asm properties ----------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVPTXMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXMCASMINFO_H
#define LLVM_LIB_TARGET_NVPTX_MCTARGETDESC_NVPTXMCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
class Triple;

class NVPTXMCAsmInfo : public MCAsmInfo {
public:
  explicit NVPTXMCAsmInfo(const Triple &TheTriple,
                          const MCTargetOptions &Options);
};
} // namespace llvm

#endif
