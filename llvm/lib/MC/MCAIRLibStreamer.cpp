//===- lib/MC/MCAIRLibStreamer.cpp - AIRLib Impl ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the object streamer for AIRLib (.metallib) files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAIRLibStreamer.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

MCStreamer *llvm::createAIRLibStreamer(MCContext &Context,
                                         std::unique_ptr<MCAsmBackend> &&MAB,
                                         std::unique_ptr<MCObjectWriter> &&OW,
                                         std::unique_ptr<MCCodeEmitter> &&CE) {
  auto *S = new MCAIRLibStreamer(Context, std::move(MAB), std::move(OW),
                                   std::move(CE));
  return S;
}
