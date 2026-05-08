//===- MCAsmStreamer.h - Base Class for Asm Streamers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCAsmBaseStreamer class, a base class for streamers
// which emits assembly text.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMSTREAMER_H
#define LLVM_MC_MCASMSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {

class MCContext;

class MCAsmBaseStreamer : public MCStreamer {
protected:
  MCAsmBaseStreamer(MCContext &Context) : MCStreamer(Context) {}
};

} // end namespace llvm

#endif // LLVM_MC_MCASMSTREAMER_H
