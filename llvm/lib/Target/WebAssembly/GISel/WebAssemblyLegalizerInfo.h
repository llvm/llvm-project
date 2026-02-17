//===- WebAssemblyLegalizerInfo.h --------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the targeting of the Machinelegalizer class for
/// WebAssembly.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_GISEL_WEBASSEMBLYMACHINELEGALIZER_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_GISEL_WEBASSEMBLYMACHINELEGALIZER_H

#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"

namespace llvm {

class WebAssemblySubtarget;

/// This class provides the information for the WebAssembly target legalizer for
/// GlobalISel.
class WebAssemblyLegalizerInfo : public LegalizerInfo {
public:
  WebAssemblyLegalizerInfo(const WebAssemblySubtarget &ST);
};
} // namespace llvm
#endif
