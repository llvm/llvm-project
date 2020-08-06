//===----- P2ELFStreamer.h - P2 Target Streamer --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_P2_ELF_STREAMER_H
#define LLVM_P2_ELF_STREAMER_H

#include "P2TargetStreamer.h"

namespace llvm {

/// A target streamer for an P2 ELF object file.
class P2ELFStreamer : public P2TargetStreamer {
public:
    P2ELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

    MCELFStreamer &getStreamer() {
        return static_cast<MCELFStreamer &>(Streamer);
    }
};

} // end namespace llvm

#endif