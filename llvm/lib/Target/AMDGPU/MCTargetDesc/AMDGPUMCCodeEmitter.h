//===-- AMDGPUCodeEmitter.h - AMDGPU Code Emitter interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// CodeEmitter interface for SI codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCCODEEMITTER_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCCODEEMITTER_H

#include "llvm/ADT/APInt.h"
#include "llvm/MC/MCCodeEmitter.h"

namespace llvm {

class MCInst;
class MCInstrInfo;
class MCOperand;
class MCSubtargetInfo;
class FeatureBitset;

class AMDGPUMCCodeEmitter : public MCCodeEmitter {
  virtual void anchor();

protected:
  const MCInstrInfo &MCII;

  AMDGPUMCCodeEmitter(const MCInstrInfo &mcii) : MCII(mcii) {}

public:
  void getBinaryCodeForInstr(const MCInst &MI, SmallVectorImpl<MCFixup> &Fixups,
                             APInt &Inst, APInt &Scratch,
                             const MCSubtargetInfo &STI) const;

  virtual void getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                 APInt &Op, SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const = 0;

  virtual void getSOPPBrEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const = 0;

  virtual void getSMEMOffsetEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const = 0;

  virtual void getSDWASrcEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                                  SmallVectorImpl<MCFixup> &Fixups,
                                  const MCSubtargetInfo &STI) const = 0;

  virtual void getSDWAVopcDstEncoding(const MCInst &MI, unsigned OpNo,
                                      APInt &Op,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const = 0;

  virtual void getAVOperandEncoding(const MCInst &MI, unsigned OpNo, APInt &Op,
                                    SmallVectorImpl<MCFixup> &Fixups,
                                    const MCSubtargetInfo &STI) const = 0;

protected:
  FeatureBitset computeAvailableFeatures(const FeatureBitset &FB) const;
  void
  verifyInstructionPredicates(const MCInst &MI,
                              const FeatureBitset &AvailableFeatures) const;
};

} // End namespace llvm

#endif
