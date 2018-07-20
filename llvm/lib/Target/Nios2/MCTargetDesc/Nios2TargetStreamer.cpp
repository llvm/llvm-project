//===-- Nios2TargetStreamer.cpp - Nios2 Target Streamer Methods -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Nios2 specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "Nios2TargetStreamer.h"

using namespace llvm;

Nios2TargetStreamer::Nios2TargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

Nios2TargetAsmStreamer::Nios2TargetAsmStreamer(MCStreamer &S,
                                               formatted_raw_ostream &OS)
    : Nios2TargetStreamer(S) {}
