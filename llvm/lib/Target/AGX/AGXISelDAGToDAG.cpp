//===-- AGXISelDAGToDAG.cpp - A DAG to DAG instruction selector for AGX ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Instruction selection for AGX: the TableGen generated matcher plus the
// complex patterns for ALU sources and memory addresses.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXISelLowering.h"
#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "agx-isel"
#define PASS_NAME "AGX DAG->DAG Pattern Instruction Selection"

namespace {

class AGXDAGToDAGISel : public SelectionDAGISel {
  const AGXSubtarget *Subtarget = nullptr;

public:
  AGXDAGToDAGISel() = delete;
  explicit AGXDAGToDAGISel(AGXTargetMachine &TM) : SelectionDAGISel(TM) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<AGXSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

private:
#include "AGXGenDAGISel.inc"

  void Select(SDNode *N) override;
  bool trySelectShift64(SDNode *N);

  SDValue getI32Imm(uint64_t Imm, const SDLoc &DL) {
    return CurDAG->getTargetConstant(Imm, DL, MVT::i32);
  }

  // Complex patterns.
  bool SelectIntSrc(SDValue N, SDValue &Src);
  bool SelectFPSrc(SDValue N, SDValue &Src, SDValue &Mods);
  bool SelectDeviceAddr(SDValue Addr, unsigned ElemLog2, SDValue &Base,
                        SDValue &Index, SDValue &Signed, SDValue &Shift);
  bool SelectDeviceAddr8(SDValue Addr, SDValue &Base, SDValue &Index,
                         SDValue &Signed, SDValue &Shift) {
    return SelectDeviceAddr(Addr, 0, Base, Index, Signed, Shift);
  }
  bool SelectDeviceAddr16(SDValue Addr, SDValue &Base, SDValue &Index,
                          SDValue &Signed, SDValue &Shift) {
    return SelectDeviceAddr(Addr, 1, Base, Index, Signed, Shift);
  }
  bool SelectDeviceAddr32(SDValue Addr, SDValue &Base, SDValue &Index,
                          SDValue &Signed, SDValue &Shift) {
    return SelectDeviceAddr(Addr, 2, Base, Index, Signed, Shift);
  }
  bool SelectLocalAddr(SDValue Addr, SDValue &Base, SDValue &Index,
                       unsigned Unit);
  bool SelectLocalAddr8(SDValue Addr, SDValue &Base, SDValue &Index) {
    return SelectLocalAddr(Addr, Base, Index, 4);
  }
  bool SelectLocalAddr16(SDValue Addr, SDValue &Base, SDValue &Index) {
    return SelectLocalAddr(Addr, Base, Index, 8);
  }
  bool SelectLocalAddr32(SDValue Addr, SDValue &Base, SDValue &Index) {
    return SelectLocalAddr(Addr, Base, Index, 16);
  }
  bool SelectStackAddr(SDValue Addr, SDValue &Offset);
};

class AGXDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;
  explicit AGXDAGToDAGISelLegacy(AGXTargetMachine &TM)
      : SelectionDAGISelLegacy(ID, std::make_unique<AGXDAGToDAGISel>(TM)) {}
};

} // namespace

char AGXDAGToDAGISelLegacy::ID = 0;

INITIALIZE_PASS(AGXDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createAGXISelDag(AGXTargetMachine &TM) {
  return new AGXDAGToDAGISelLegacy(TM);
}

void AGXDAGToDAGISel::Select(SDNode *N) {
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;
  }
  switch (N->getOpcode()) {
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    // Shifts of extended 32-bit values by up to 4 are selected as iadd by
    // the generated matcher.
    if (N->getOpcode() == ISD::SHL &&
        (N->getOperand(0).getOpcode() == ISD::ZERO_EXTEND ||
         N->getOperand(0).getOpcode() == ISD::SIGN_EXTEND) &&
        isa<ConstantSDNode>(N->getOperand(1)) &&
        N->getConstantOperandVal(1) <= 4)
      break;
    if (trySelectShift64(N))
      return;
    break;
  default:
    break;
  }
  SelectCode(N);
}

/// Integer immediates 0-255 are encoded in the instruction.
bool AGXDAGToDAGISel::SelectIntSrc(SDValue N, SDValue &Src) {
  if (auto *C = dyn_cast<ConstantSDNode>(N)) {
    if (C->getZExtValue() <= 0xff) {
      Src = CurDAG->getTargetConstant(C->getZExtValue(), SDLoc(N),
                                      N.getValueType());
      return true;
    }
  }
  Src = N;
  return true;
}

/// Floating point sources fold neg and abs into modifiers, and constants
/// representable as 8-bit floats into the instruction.
bool AGXDAGToDAGISel::SelectFPSrc(SDValue N, SDValue &Src, SDValue &Mods) {
  SDLoc DL(N);
  unsigned M = 0;
  while (true) {
    if (N.getOpcode() == ISD::FNEG) {
      M ^= AGX::SRC_MOD_NEG;
      N = N.getOperand(0);
    } else if (N.getOpcode() == ISD::FABS) {
      // |-x| = |x|, and an outer negation applies after the absolute value.
      M |= AGX::SRC_MOD_ABS;
      N = N.getOperand(0);
    } else {
      break;
    }
  }

  if (auto *C = dyn_cast<ConstantFPSDNode>(N)) {
    APFloat V = C->getValueAPF();
    if (M & AGX::SRC_MOD_ABS)
      V.clearSign();
    if (M & AGX::SRC_MOD_NEG)
      V.changeSign();
    bool LosesInfo = false;
    V.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
              &LosesInfo);
    if (!LosesInfo) {
      if (std::optional<unsigned> Enc = AGX::encodeFPImm8(V.convertToDouble())) {
        Src = CurDAG->getTargetConstant(*Enc, DL, MVT::i32);
        Mods = getI32Imm(0, DL);
        return true;
      }
    }
  }

  Src = N;
  Mods = getI32Imm(M, DL);
  return true;
}

/// 64-bit shifts by a constant: shift both halves, combining them with a
/// funnel shift (extr).
bool AGXDAGToDAGISel::trySelectShift64(SDNode *N) {
  auto *C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!C || N->getValueType(0) != MVT::i64)
    return false;
  SDLoc DL(N);
  unsigned Opc = N->getOpcode();
  uint64_t S = C->getZExtValue() & 63;
  SDValue X = N->getOperand(0);
  SDValue Lo = CurDAG->getTargetExtractSubreg(AGX::sub0, DL, MVT::i32, X);
  SDValue Hi = CurDAG->getTargetExtractSubreg(AGX::sub1, DL, MVT::i32, X);
  SDValue Zero = getI32Imm(0, DL);

  // Machine node builders mirroring the 32-bit shift patterns.
  auto Shl = [&](SDValue V, uint64_t A) {
    return SDValue(CurDAG->getMachineNode(AGX::BFI_I32, DL, MVT::i32,
                                          {Zero, Zero, Zero, V, Zero,
                                           getI32Imm(A, DL), Zero}),
                   0);
  };
  auto Lsr = [&](SDValue V, uint64_t A) {
    return SDValue(CurDAG->getMachineNode(AGX::BFEIL_I32, DL, MVT::i32,
                                          {Zero, Zero, Zero, V, Zero,
                                           getI32Imm(A, DL), Zero}),
                   0);
  };
  auto Asr = [&](SDValue V, uint64_t A) {
    return SDValue(CurDAG->getMachineNode(AGX::ASR_I32, DL, MVT::i32,
                                          {Zero, V, Zero, getI32Imm(A, DL)}),
                   0);
  };
  // (B:A) >> A, the low 32 bits.
  auto Extr = [&](SDValue A, SDValue B, uint64_t Amt) {
    return SDValue(CurDAG->getMachineNode(AGX::EXTR_I32, DL, MVT::i32,
                                          {Zero, A, Zero, B, Zero,
                                           getI32Imm(Amt, DL), Zero}),
                   0);
  };
  auto Imm0 = [&]() {
    return SDValue(
        CurDAG->getMachineNode(AGX::MOV_IMM32, DL, MVT::i32, Zero), 0);
  };

  SDValue NewLo, NewHi;
  if (S == 0) {
    ReplaceNode(N, X.getNode());
    return true;
  }
  switch (Opc) {
  case ISD::SHL:
    if (S >= 32) {
      NewLo = Imm0();
      NewHi = Shl(Lo, S - 32);
    } else {
      NewLo = Shl(Lo, S);
      NewHi = Extr(Lo, Hi, 32 - S);
    }
    break;
  case ISD::SRL:
    if (S >= 32) {
      NewLo = Lsr(Hi, S - 32);
      NewHi = Imm0();
    } else {
      NewLo = Extr(Lo, Hi, S);
      NewHi = Lsr(Hi, S);
    }
    break;
  case ISD::SRA:
    if (S >= 32) {
      NewLo = Asr(Hi, S - 32);
      NewHi = Asr(Hi, 31);
    } else {
      NewLo = Extr(Lo, Hi, S);
      NewHi = Asr(Hi, S);
    }
    break;
  default:
    return false;
  }
  SDValue Ops[] = {
      CurDAG->getTargetConstant(AGX::GPR64RegClassID, DL, MVT::i32), NewLo,
      CurDAG->getTargetConstant(AGX::sub0, DL, MVT::i32), NewHi,
      CurDAG->getTargetConstant(AGX::sub1, DL, MVT::i32)};
  ReplaceNode(N, CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                        MVT::i64, Ops));
  return true;
}

/// Device addresses are base + (index << (log2(element size) + shift)), with
/// a 32-bit index that is zero or sign extended, or a signed 16-bit
/// immediate index.
bool AGXDAGToDAGISel::SelectDeviceAddr(SDValue Addr, unsigned ElemLog2,
                                       SDValue &Base, SDValue &Index,
                                       SDValue &Signed, SDValue &Shift) {
  SDLoc DL(Addr);
  Signed = getI32Imm(0, DL);
  Shift = getI32Imm(0, DL);

  if (Addr.getOpcode() == ISD::ADD ||
      (Addr.getOpcode() == ISD::OR && Addr->getFlags().hasDisjoint())) {
    for (unsigned I = 0; I < 2; ++I) {
      SDValue B = Addr.getOperand(I);
      SDValue O = Addr.getOperand(1 - I);

      // Constant byte offset.
      if (auto *C = dyn_cast<ConstantSDNode>(O)) {
        int64_t Off = C->getSExtValue();
        if (Off % (int64_t(1) << ElemLog2) == 0 &&
            isInt<16>(Off >> ElemLog2)) {
          Base = B;
          Index = CurDAG->getSignedTargetConstant(Off >> ElemLog2, DL, MVT::i32);
          return true;
        }
        continue;
      }

      // Scaled, extended 32-bit index.
      unsigned Scale = 0;
      if (O.getOpcode() == ISD::SHL) {
        auto *C = dyn_cast<ConstantSDNode>(O.getOperand(1));
        if (!C)
          continue;
        Scale = C->getZExtValue();
        O = O.getOperand(0);
      } else if (O.getOpcode() == ISD::MUL) {
        auto *C = dyn_cast<ConstantSDNode>(O.getOperand(1));
        if (!C || !isPowerOf2_64(C->getZExtValue()))
          continue;
        Scale = Log2_64(C->getZExtValue());
        O = O.getOperand(0);
      }
      // The shift field is 0 to 2: applegpu notes that 3 may act as 2.
      if (Scale < ElemLog2 || Scale > ElemLog2 + 2)
        continue;
      bool IsSExt = O.getOpcode() == ISD::SIGN_EXTEND;
      if (!IsSExt && O.getOpcode() != ISD::ZERO_EXTEND)
        continue;
      if (O.getOperand(0).getValueType() != MVT::i32)
        continue;
      Base = B;
      Index = O.getOperand(0);
      // The index signedness field is 1 for zero extension.
      Signed = getI32Imm(!IsSExt, DL);
      Shift = getI32Imm(Scale - ElemLog2, DL);
      return true;
    }
  }

  Base = Addr;
  Index = CurDAG->getTargetConstant(0, DL, MVT::i32);
  return true;
}

/// Threadgroup addresses are 16-bit byte offsets in the base register. The
/// index counts four-element vectors, Unit bytes each (the address is base +
/// 4 * element size * index), so constant offsets that are multiples of Unit
/// go in the immediate index.
bool AGXDAGToDAGISel::SelectLocalAddr(SDValue Addr, SDValue &Base,
                                      SDValue &Index, unsigned Unit) {
  SDLoc DL(Addr);
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    int64_t Offset = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    SDValue B = Addr.getOperand(0);
    if (Offset > 0 && Offset % Unit == 0 && isUInt<15>(Offset / Unit)) {
      Base = isNullConstant(B)
                 ? CurDAG->getTargetConstant(0, DL, MVT::i16)
                 : CurDAG->getTargetExtractSubreg(AGX::lo16, DL, MVT::i16, B);
      Index = CurDAG->getTargetConstant(Offset / Unit, DL, MVT::i16);
      return true;
    }
    // base + (Offset % Unit) is already computed for another access: reuse
    // it with the rest of the offset in the index.
    if (Offset > int64_t(Unit) && isUInt<15>(Offset / Unit)) {
      SDValue Rem = CurDAG->getConstant(Offset % Unit, DL, MVT::i32);
      if (SDNode *N = CurDAG->getNodeIfExists(
              ISD::ADD, CurDAG->getVTList(MVT::i32), {B, Rem})) {
        Base = CurDAG->getTargetExtractSubreg(AGX::lo16, DL, MVT::i16,
                                              SDValue(N, 0));
        Index = CurDAG->getTargetConstant(Offset / Unit, DL, MVT::i16);
        return true;
      }
    }
  }
  if (isNullConstant(Addr))
    Base = CurDAG->getTargetConstant(0, DL, MVT::i16);
  else
    Base = CurDAG->getTargetExtractSubreg(AGX::lo16, DL, MVT::i16, Addr);
  Index = CurDAG->getTargetConstant(0, DL, MVT::i16);
  return true;
}

/// Stack addresses are byte offsets: a frame index (replaced by its offset
/// after frame lowering), a 16-bit immediate or a register.
bool AGXDAGToDAGISel::SelectStackAddr(SDValue Addr, SDValue &Offset) {
  SDLoc DL(Addr);
  if (auto *FI = dyn_cast<FrameIndexSDNode>(Addr)) {
    Offset = CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
    return true;
  }
  if (auto *C = dyn_cast<ConstantSDNode>(Addr)) {
    if (isInt<16>(C->getSExtValue())) {
      Offset = CurDAG->getTargetConstant(C->getSExtValue(), DL, MVT::i32);
      return true;
    }
  }
  Offset = Addr;
  return true;
}
