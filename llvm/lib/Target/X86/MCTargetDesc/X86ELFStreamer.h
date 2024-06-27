//===-- X86ELFStreamer.h - ELF Streamer for X86 -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_MCTARGETDESC_X86ELFSTREAMER_H
#define LLVM_LIB_TARGET_X86_MCTARGETDESC_X86ELFSTREAMER_H

#include "llvm/MC/MCELFStreamer.h"

namespace llvm {

MCStreamer *createX86ELFStreamer(const Triple &T, MCContext &Context,
                                 std::unique_ptr<MCAsmBackend> &&MAB,
                                 std::unique_ptr<MCObjectWriter> &&MOW,
                                 std::unique_ptr<MCCodeEmitter> &&MCE);
}

#endif
