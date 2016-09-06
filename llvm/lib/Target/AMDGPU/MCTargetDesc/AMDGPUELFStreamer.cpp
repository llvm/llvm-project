//===-------- AMDGPUELFStreamer.cpp - ELF Object Output -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUELFStreamer.h"
#include "Utils/AMDGPUBaseInfo.h"

using namespace llvm;

MCELFStreamer *llvm::createAMDGPUELFStreamer(MCContext &Context,
                                           MCAsmBackend &MAB,
                                           raw_pwrite_stream &OS,
                                           MCCodeEmitter *Emitter,
                                           bool RelaxAll) {
  return new AMDGPUELFStreamer(Context, MAB, OS, Emitter);
}
