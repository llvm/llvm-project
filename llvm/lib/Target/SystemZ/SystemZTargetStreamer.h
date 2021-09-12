//=- SystemZTargetStreamer.h - SystemZ Target Streamer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETSTREAMER_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETSTREAMER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {

class SystemZTargetStreamer : public MCTargetStreamer {
public:
  SystemZTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

  virtual void emitMachine(StringRef CPU) = 0;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETSTREAMER_H
