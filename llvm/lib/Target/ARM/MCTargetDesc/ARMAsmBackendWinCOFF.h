//===-- ARMAsmBackendWinCOFF.h - ARM Asm Backend WinCOFF --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMASMBACKENDWINCOFF_H
#define LLVM_LIB_TARGET_ARM_ARMASMBACKENDWINCOFF_H

#include "ARMAsmBackend.h"
#include "llvm/MC/MCObjectWriter.h"
using namespace llvm;

namespace {
class ARMAsmBackendWinCOFF : public ARMAsmBackend {
public:
  ARMAsmBackendWinCOFF(const Target &T, const MCSubtargetInfo &STI)
      : ARMAsmBackend(T, STI, true) {}
  std::unique_ptr<MCObjectWriter>
  createObjectWriter(raw_pwrite_stream &OS) const override {
    return createARMWinCOFFObjectWriter(OS, /*Is64Bit=*/false);
  }
};
}

#endif
