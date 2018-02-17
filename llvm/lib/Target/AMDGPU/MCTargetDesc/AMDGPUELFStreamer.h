//===-------- AMDGPUELFStreamer.h - ELF Object Output -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a custom MCELFStreamer which allows us to insert some hooks before
// emitting data into an actual object file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUELFSTREAMER_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUELFSTREAMER_H

#include "llvm/MC/MCELFStreamer.h"

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCSubtargetInfo;

class AMDGPUELFStreamer : public MCELFStreamer {
public:
  AMDGPUELFStreamer(const Triple &T, MCContext &Context,
                    std::unique_ptr<MCAsmBackend> MAB, raw_pwrite_stream &OS,
                    std::unique_ptr<MCCodeEmitter> Emitter)
      : MCELFStreamer(Context, std::move(MAB), OS, std::move(Emitter)) {}
};

MCELFStreamer *createAMDGPUELFStreamer(const Triple &T, MCContext &Context,
                                       std::unique_ptr<MCAsmBackend> MAB,
                                       raw_pwrite_stream &OS,
                                       std::unique_ptr<MCCodeEmitter> Emitter,
                                       bool RelaxAll);
} // namespace llvm.

#endif
