//===- lib/MC/MCSPIRVStreamer.cpp - SPIR-V Object Output ------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits SPIR-V .o object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSPIRVStreamer.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

MCStreamer *llvm::createSPIRVStreamer(MCContext &Context,
                                      std::unique_ptr<MCAsmBackend> &&MAB,
                                      std::unique_ptr<MCObjectWriter> &&OW,
                                      std::unique_ptr<MCCodeEmitter> &&CE) {
  MCSPIRVStreamer *S = new MCSPIRVStreamer(Context, std::move(MAB),
                                           std::move(OW), std::move(CE));
  return S;
}
