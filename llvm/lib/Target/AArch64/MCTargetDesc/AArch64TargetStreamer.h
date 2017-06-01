//===-- AArch64TargetStreamer.h - AArch64 Target Streamer ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64TARGETSTREAMER_H
#define LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64TARGETSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {

class AArch64TargetStreamer : public MCTargetStreamer {
public:
  AArch64TargetStreamer(MCStreamer &S);
  ~AArch64TargetStreamer() override;

  void finish() override;

  /// Callback used to implement the ldr= pseudo.
  /// Add a new entry to the constant pool for the current section and return an
  /// MCExpr that can be used to refer to the constant pool location.
  const MCExpr *addConstantPoolEntry(const MCExpr *, unsigned Size, SMLoc Loc);

  /// Callback used to implemnt the .ltorg directive.
  /// Emit contents of constant pool for the current section.
  void emitCurrentConstantPool();

  /// Callback used to implement the .inst directive.
  virtual void emitInst(uint32_t Inst);

private:
  std::unique_ptr<AssemblerConstantPools> ConstantPools;
};

} // end namespace llvm

#endif
