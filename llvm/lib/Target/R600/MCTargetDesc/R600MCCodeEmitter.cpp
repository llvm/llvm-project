//===- R600MCCodeEmitter.cpp - Code Emitter for R600->Cayman GPU families -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// \brief The R600 code emitter produces machine code that can be executed
/// directly on the GPU device.
//
//===----------------------------------------------------------------------===//

#include "R600Defines.h"
#include "MCTargetDesc/AMDGPUMCCodeEmitter.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <stdio.h>

using namespace llvm;

namespace {

class R600MCCodeEmitter : public AMDGPUMCCodeEmitter {
  R600MCCodeEmitter(const R600MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const R600MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  const MCSubtargetInfo &STI;

public:

  R600MCCodeEmitter(const MCInstrInfo &mcii, const MCRegisterInfo &mri,
                    const MCSubtargetInfo &sti)
    : MCII(mcii), MRI(mri), STI(sti) { }

  /// \brief Encode the instruction and write it to the OS.
  virtual void EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups) const;

  /// \returns the encoding for an MCOperand.
  virtual uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                     SmallVectorImpl<MCFixup> &Fixups) const;
private:

  void EmitByte(unsigned int byte, raw_ostream &OS) const;

  void Emit(uint32_t value, raw_ostream &OS) const;
  void Emit(uint64_t value, raw_ostream &OS) const;

  unsigned getHWRegChan(unsigned reg) const;
  unsigned getHWReg(unsigned regNo) const;

};

} // End anonymous namespace

enum RegElement {
  ELEMENT_X = 0,
  ELEMENT_Y,
  ELEMENT_Z,
  ELEMENT_W
};

enum FCInstr {
  FC_IF_PREDICATE = 0,
  FC_ELSE,
  FC_ENDIF,
  FC_BGNLOOP,
  FC_ENDLOOP,
  FC_BREAK_PREDICATE,
  FC_CONTINUE
};

enum TextureTypes {
  TEXTURE_1D = 1,
  TEXTURE_2D,
  TEXTURE_3D,
  TEXTURE_CUBE,
  TEXTURE_RECT,
  TEXTURE_SHADOW1D,
  TEXTURE_SHADOW2D,
  TEXTURE_SHADOWRECT,
  TEXTURE_1D_ARRAY,
  TEXTURE_2D_ARRAY,
  TEXTURE_SHADOW1D_ARRAY,
  TEXTURE_SHADOW2D_ARRAY
};

MCCodeEmitter *llvm::createR600MCCodeEmitter(const MCInstrInfo &MCII,
                                           const MCRegisterInfo &MRI,
                                           const MCSubtargetInfo &STI) {
  return new R600MCCodeEmitter(MCII, MRI, STI);
}

void R600MCCodeEmitter::EncodeInstruction(const MCInst &MI, raw_ostream &OS,
                                       SmallVectorImpl<MCFixup> &Fixups) const {
  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  if (MI.getOpcode() == AMDGPU::RETURN ||
    MI.getOpcode() == AMDGPU::FETCH_CLAUSE ||
    MI.getOpcode() == AMDGPU::ALU_CLAUSE ||
    MI.getOpcode() == AMDGPU::BUNDLE ||
    MI.getOpcode() == AMDGPU::KILL) {
    return;
  } else if (IS_VTX(Desc)) {
    uint64_t InstWord01 = getBinaryCodeForInstr(MI, Fixups);
    uint32_t InstWord2 = MI.getOperand(2).getImm(); // Offset
    InstWord2 |= 1 << 19;

    Emit(InstWord01, OS);
    Emit(InstWord2, OS);
    Emit((u_int32_t) 0, OS);
  } else if (IS_TEX(Desc)) {
    unsigned Opcode = MI.getOpcode();
    bool HasOffsets = (Opcode == AMDGPU::TEX_LD);
    unsigned OpOffset = HasOffsets ? 3 : 0;
    int64_t Sampler = MI.getOperand(OpOffset + 3).getImm();
    int64_t TextureType = MI.getOperand(OpOffset + 4).getImm();

    uint32_t SrcSelect[4] = {0, 1, 2, 3};
    uint32_t Offsets[3] = {0, 0, 0};
    uint64_t CoordType[4] = {1, 1, 1, 1};

    if (HasOffsets)
      for (unsigned i = 0; i < 3; i++) {
        int SignedOffset = MI.getOperand(i + 2).getImm();
        Offsets[i] = (SignedOffset & 0x1F);
      }

    if (TextureType == TEXTURE_RECT ||
        TextureType == TEXTURE_SHADOWRECT) {
      CoordType[ELEMENT_X] = 0;
      CoordType[ELEMENT_Y] = 0;
    }

    if (TextureType == TEXTURE_1D_ARRAY ||
        TextureType == TEXTURE_SHADOW1D_ARRAY) {
      if (Opcode == AMDGPU::TEX_SAMPLE_C_L ||
          Opcode == AMDGPU::TEX_SAMPLE_C_LB) {
        CoordType[ELEMENT_Y] = 0;
      } else {
        CoordType[ELEMENT_Z] = 0;
        SrcSelect[ELEMENT_Z] = ELEMENT_Y;
      }
    } else if (TextureType == TEXTURE_2D_ARRAY ||
        TextureType == TEXTURE_SHADOW2D_ARRAY) {
      CoordType[ELEMENT_Z] = 0;
    }


    if ((TextureType == TEXTURE_SHADOW1D ||
        TextureType == TEXTURE_SHADOW2D ||
        TextureType == TEXTURE_SHADOWRECT ||
        TextureType == TEXTURE_SHADOW1D_ARRAY) &&
        Opcode != AMDGPU::TEX_SAMPLE_C_L &&
        Opcode != AMDGPU::TEX_SAMPLE_C_LB) {
      SrcSelect[ELEMENT_W] = ELEMENT_Z;
    }

    uint64_t Word01 = getBinaryCodeForInstr(MI, Fixups) |
        CoordType[ELEMENT_X] << 60 | CoordType[ELEMENT_Y] << 61 |
        CoordType[ELEMENT_Z] << 62 | CoordType[ELEMENT_W] << 63;
    uint32_t Word2 = Sampler << 15 | SrcSelect[ELEMENT_X] << 20 |
        SrcSelect[ELEMENT_Y] << 23 | SrcSelect[ELEMENT_Z] << 26 |
        SrcSelect[ELEMENT_W] << 29 | Offsets[0] << 0 | Offsets[1] << 5 |
        Offsets[2] << 10;

    Emit(Word01, OS);
    Emit(Word2, OS);
    Emit((u_int32_t) 0, OS);
  } else {
    uint64_t Inst = getBinaryCodeForInstr(MI, Fixups);
    if ((STI.getFeatureBits() & AMDGPU::FeatureR600ALUInst) &&
       ((Desc.TSFlags & R600_InstFlag::OP1) ||
         Desc.TSFlags & R600_InstFlag::OP2)) {
      uint64_t ISAOpCode = Inst & (0x3FFULL << 39);
      Inst &= ~(0x3FFULL << 39);
      Inst |= ISAOpCode << 1;
    }
    Emit(Inst, OS);
  }
}

void R600MCCodeEmitter::EmitByte(unsigned int Byte, raw_ostream &OS) const {
  OS.write((uint8_t) Byte & 0xff);
}

void R600MCCodeEmitter::Emit(uint32_t Value, raw_ostream &OS) const {
  for (unsigned i = 0; i < 4; i++) {
    OS.write((uint8_t) ((Value >> (8 * i)) & 0xff));
  }
}

void R600MCCodeEmitter::Emit(uint64_t Value, raw_ostream &OS) const {
  for (unsigned i = 0; i < 8; i++) {
    EmitByte((Value >> (8 * i)) & 0xff, OS);
  }
}

unsigned R600MCCodeEmitter::getHWRegChan(unsigned reg) const {
  return MRI.getEncodingValue(reg) >> HW_CHAN_SHIFT;
}

unsigned R600MCCodeEmitter::getHWReg(unsigned RegNo) const {
  return MRI.getEncodingValue(RegNo) & HW_REG_MASK;
}

uint64_t R600MCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                              const MCOperand &MO,
                                        SmallVectorImpl<MCFixup> &Fixup) const {
  if (MO.isReg()) {
    if (HAS_NATIVE_OPERANDS(MCII.get(MI.getOpcode()).TSFlags)) {
      return MRI.getEncodingValue(MO.getReg());
    } else {
      return getHWReg(MO.getReg());
    }
  } else if (MO.isImm()) {
    return MO.getImm();
  } else {
    assert(0);
    return 0;
  }
}

#include "AMDGPUGenMCCodeEmitter.inc"
