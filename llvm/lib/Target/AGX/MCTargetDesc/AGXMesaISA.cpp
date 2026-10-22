//===-- AGXMesaISA.cpp - Mesa's AGX instruction forms ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The engine that checks an instruction against the generated forms. It
// follows check_form and check_field in llvm/utils/agx-mesa-isa/mesa_isa.py.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AGXMesaISA.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Format.h"
#include <cstdint>

using namespace llvm;

namespace {

struct MesaSet {
  uint16_t Start, Count;
};

struct MesaField {
  uint8_t Rule;
  int8_t Ranges[4][2];
  uint16_t P[3];
};

struct MesaForm {
  const char *Name;
  uint8_t Length;
  uint64_t Mask[2], Value[2], LongMask[2];
  uint16_t First;
  uint8_t Count;
  uint32_t Same;
  uint8_t Gens;
  uint8_t Dst;
};

#define AGX_MESA_ISA_SETS
#define AGX_MESA_ISA_FIELDS
#define AGX_MESA_ISA_FORMS
#include "AGXMesaISA.inc"

enum Rule {
  Any,
  Set,
  Range,
  Even,
  ALUDst,
  ALUSrc,
  MemReg,
  MemBase,
  MemIndex,
  LocalBase,
  LocalIndex,
  ZeroIndex,
  AtomicDst,
  LocalAtomicDst,
  TileReg,
  IterDst,
  SampleMaskSrc,
  TexExt,
};

enum SrcKind { KInt, KIAdd, KFloat, KHalf, KCmpsel };
constexpr unsigned SrcWidth[] = {10, 11, 12, 11, 9};

enum : unsigned { Size16 = 1, Size32 = 2, Size64 = 4 };
enum OpType { None, Imm, Uniform, Reg };

struct Operand {
  OpType Type = None;
  unsigned Size = 0;
};

struct Bits {
  uint64_t W[2];
  bool get(unsigned I) const { return (W[I / 64] >> (I % 64)) & 1; }
};

uint64_t extract(const Bits &B, const MesaField &F) {
  uint64_t V = 0;
  unsigned Shift = 0;
  for (const auto &R : F.Ranges) {
    if (R[1] < R[0])
      break;
    for (int I = R[0]; I <= R[1]; ++I)
      V |= uint64_t(B.get(I)) << Shift++;
  }
  return V;
}

bool inSet(unsigned Index, uint64_t V) {
  const MesaSet &S = MesaSets[Index];
  for (unsigned I = 0; I < S.Count; ++I)
    if (MesaSetValues[S.Start + I] == V)
      return true;
  return false;
}

bool aligned(unsigned Value, unsigned Size) {
  if (Size == Size32)
    return Value % 2 == 0;
  if (Size == Size64)
    return Value % 4 == 0;
  return true;
}

/// Decodes an ALU source as agx_pack_alu_src / agx_pack_cmpsel_src encode
/// it. Returns false for an encoding Mesa never produces.
bool decodeALUSrc(unsigned Kind, uint64_t V, OpType &Type, unsigned &Size,
                  unsigned &Value) {
  unsigned Width = SrcWidth[Kind];
  uint64_t Short = V & ((1u << Width) - 1), Ext = V >> Width;
  unsigned V6 = Short & 63, B67 = (Short >> 6) & 3, B8 = (Short >> 8) & 1;
  Value = V6 | (Ext << 6);
  Size = 0;
  if (Kind == KCmpsel) {
    if (B8 && B67 == 0)
      Type = Imm;
    else if (B8 && (B67 >> 1)) {
      Type = Uniform;
      Value |= (B67 & 1) << 8;
    } else if (!B8 && B67)
      Type = Reg;
    else
      return false;
    return true;
  }
  unsigned B9 = Kind == KHalf ? 0 : (Short >> 9) & 1;
  if (B67 == 0 && !B8 && !B9) {
    Type = Imm;
    return true;
  }
  if (B8 && !B9) {
    Type = Uniform;
    Size = (B67 >> 1) ? Size32 : Size16;
    Value |= (B67 & 1) << 8;
    return true;
  }
  if (B67 == 0)
    return false;
  Type = Reg;
  Size = B9 ? (B8 ? Size64 : Size32) : Size16;
  return true;
}

const char *checkField(const MesaForm &Form, unsigned Index, uint64_t V,
                       Operand *Ops) {
  const MesaField &F = MesaFields[Form.First + Index];
  switch (F.Rule) {
  case Any:
  case TileReg:
  case IterDst:
    return nullptr;
  case Set:
    return inSet(F.P[0], V) ? nullptr : "field value";
  case Range:
    return V >= F.P[0] && V <= F.P[1] ? nullptr : "field range";
  case Even:
    return V % 2 == 0 && V < 256 ? nullptr : "register";
  case ALUDst: {
    unsigned Cache = V & 1, S32 = (V >> 1) & 1, Value = V >> 2, Size = Size16;
    if (S32) {
      Size = (Value & 1) ? Size64 : Size32;
      Value &= ~1u;
      if (Size == Size64 && Value % 4)
        return "destination alignment";
    }
    if (!(Size & F.P[0]))
      return "destination size";
    if (Cache && !F.P[1])
      return "destination cache hint";
    Ops[Index] = {Reg, Size};
    return nullptr;
  }
  case ALUSrc: {
    unsigned Kind = F.P[0], Size, Value;
    OpType Type;
    if (!decodeALUSrc(Kind, V, Type, Size, Value))
      return "source encoding";
    if (Kind == KCmpsel) {
      Size = Form.Dst < Form.Count ? Ops[Form.Dst].Size : Size32;
      if (Type == Reg && !aligned(Value, Size))
        return "source alignment";
      Ops[Index] = {Type, Size};
      return nullptr;
    }
    if (F.P[2]) {
      if (Type != Imm)
        return "source must be an immediate";
      if (!inSet(F.P[2] - 1, Value))
        return "immediate source value";
    }
    if ((Type == Reg || Type == Uniform) && !(Size & F.P[1]))
      return "source size";
    if (Type == Reg && !aligned(Value, Size))
      return "source alignment";
    uint64_t Short = V & ((1u << SrcWidth[Kind]) - 1);
    if (Kind == KIAdd && ((Short >> 10) & 1) && (Type == Imm || Size == Size64))
      return "source sign extension";
    Ops[Index] = {Type, Size};
    return nullptr;
  }
  case MemReg:
    return !(V >> 8) || (V & 0xff) % 2 == 0 ? nullptr : "register alignment";
  case MemBase:
    return (V & 0xff) % 2 == 0 ? nullptr : "base alignment";
  case MemIndex:
    if (V >> 16)
      return nullptr;
    return (V & 0xffff) % 2 == 0 && (V & 0xffff) < 0x100 ? nullptr
                                                          : "index register";
  case LocalBase:
    return (V >> 8) == 2 && (V & 0xff) ? "local base immediate" : nullptr;
  case LocalIndex:
    return (V >> 16) || (V & 0xffff) < 0x100 ? nullptr : "local index register";
  case ZeroIndex:
    return V == (1u << 16) ? nullptr : "local atomic index";
  case AtomicDst:
    if (!(V >> 8))
      return (V & 0xff) ? "atomic destination" : nullptr;
    return (V & 0xff) % 2 == 0 ? nullptr : "atomic destination alignment";
  case LocalAtomicDst:
    if (!(V >> 9))
      return (V & 0x1ff) ? "atomic destination" : nullptr;
    return !((V >> 8) & 1) || (V & 0xff) % 2 == 0
               ? nullptr
               : "atomic destination alignment";
  case SampleMaskSrc:
    return (V >> 6) & 1 ? "sample mask source" : nullptr;
  case TexExt:
    return (V >> 5) & 1 ? "texture kill bit" : nullptr;
  }
  return "rule";
}

const char *checkForm(const MesaForm &Form, const Bits &B, unsigned GenBit) {
  if ((B.W[0] & Form.Mask[0]) != Form.Value[0] ||
      (B.W[1] & Form.Mask[1]) != Form.Value[1])
    return "fixed bits";
  Operand Ops[24];
  for (unsigned I = 0; I < Form.Count; ++I)
    if (const char *Why =
            checkField(Form, I, extract(B, MesaFields[Form.First + I]), Ops))
      return Why;
  if (Form.Same) {
    unsigned Size = 0;
    for (unsigned I = 0; I < Form.Count; ++I) {
      if (!(Form.Same & (1u << I)) || Ops[I].Type == Imm || !Ops[I].Size)
        continue;
      if (Size && Size != Ops[I].Size)
        return "operand sizes differ";
      Size = Ops[I].Size;
    }
  }
  if ((Form.LongMask[0] | Form.LongMask[1]) &&
      !(B.W[0] & Form.LongMask[0]) && !(B.W[1] & Form.LongMask[1]))
    return "long form not needed";
  // Last, so that a field Mesa never packs is reported first.
  if (!(Form.Gens & GenBit))
    return "generation";
  return nullptr;
}

} // namespace

std::optional<AGX::MesaGen> AGX::getMesaGen(const MCSubtargetInfo &STI) {
  bool MultiDie = STI.hasFeature(AGX::FeatureMultiDieCoherency);
  if (STI.hasFeature(AGX::FeatureG13))
    return MultiDie ? MesaGen::G13X : MesaGen::G13;
  if (STI.hasFeature(AGX::FeatureG14))
    return MultiDie ? MesaGen::G14D : MesaGen::G14;
  return std::nullopt;
}

bool AGX::isMesaInstruction(ArrayRef<uint8_t> Bytes, MesaGen Gen,
                            StringRef *FormName, std::string *Why) {
  Bits B = {{0, 0}};
  for (unsigned I = 0; I < Bytes.size() && I < 16; ++I)
    B.W[I / 8] |= uint64_t(Bytes[I]) << (8 * (I % 8));
  unsigned GenBit = 1u << unsigned(Gen);
  const char *Closest = nullptr, *ClosestWhy = nullptr;
  for (const MesaForm &Form : MesaForms) {
    if (Form.Length != Bytes.size())
      continue;
    const char *R = checkForm(Form, B, GenBit);
    if (!R) {
      if (FormName)
        *FormName = Form.Name;
      return true;
    }
    // Report the first form whose fixed bits match: it has the same opcode.
    if (!Closest && R != StringRef("fixed bits") &&
        R != StringRef("generation")) {
      Closest = Form.Name;
      ClosestWhy = R;
    }
    if (!Closest && R == StringRef("generation")) {
      Closest = Form.Name;
      ClosestWhy = "not emitted for this GPU";
    }
  }
  if (Why) {
    if (Closest)
      *Why = std::string(Closest) + ": " + ClosestWhy;
    else
      *Why = "no Mesa instruction has this opcode and length";
  }
  return false;
}
