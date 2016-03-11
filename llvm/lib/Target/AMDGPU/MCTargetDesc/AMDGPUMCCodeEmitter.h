//===-- AMDGPUCodeEmitter.h - AMDGPU Code Emitter interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief CodeEmitter interface for R600 and SI codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCCODEEMITTER_H
#define LLVM_LIB_TARGET_AMDGPU_MCTARGETDESC_AMDGPUMCCODEEMITTER_H

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCInst;
class MCOperand;
class MCSubtargetInfo;

class AMDGPUMCCodeEmitter : public MCCodeEmitter {
  virtual void anchor();
public:

  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  virtual uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
    return 0;
  }

  virtual unsigned getSOPPBrEncoding(const MCInst &MI, unsigned OpNo,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
    return 0;
  }
};

} // End namespace llvm

#endif
