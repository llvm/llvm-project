//===-- MetalInstrInfo.h - Define InstrInfo for Metal -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Metal specific subclass of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALINSTRINFO_H
#define LLVM_LIB_TARGET_METAL_METALINSTRINFO_H

#include "MetalRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "MetalGenInstrInfo.inc"

namespace llvm {
class MetalSubtarget;

struct MetalInstrInfo : public MetalGenInstrInfo {
  const MetalRegisterInfo RI;
  explicit MetalInstrInfo(const MetalSubtarget &STI);
  const MetalRegisterInfo &getRegisterInfo() const { return RI; }
  ~MetalInstrInfo() override;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALINSTRINFO_H
