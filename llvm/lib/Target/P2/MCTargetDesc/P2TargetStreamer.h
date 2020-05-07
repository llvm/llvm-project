//===-- P2TargetStreamer.h - P2 Target Streamer ------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2TARGETSTREAMER_H
#define LLVM_LIB_TARGET_P2_P2TARGETSTREAMER_H

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {

    class P2TargetStreamer : public MCTargetStreamer {
        public:
            P2TargetStreamer(MCStreamer &S);
    };

    // This part is for ascii assembly output
    class P2TargetAsmStreamer : public P2TargetStreamer {

        public:
            P2TargetAsmStreamer(MCStreamer &S);
    };

}

#endif

