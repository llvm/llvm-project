//===---- RISCVISelDAGToDAG.h - A dag to dag inst selector for RISC-V -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the RISC-V target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVISELDAGTODAG_H
#define LLVM_LIB_TARGET_RISCV_RISCVISELDAGTODAG_H

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Support/KnownBits.h"

// RISC-V specific code to select RISC-V machine instructions for
// SelectionDAG operations.
namespace llvm {
class RISCVDAGToDAGISel : public SelectionDAGISel {
  const RISCVSubtarget *Subtarget = nullptr;

public:
  RISCVDAGToDAGISel() = delete;

  explicit RISCVDAGToDAGISel(RISCVTargetMachine &TargetMachine,
                             CodeGenOptLevel OptLevel)
      : SelectionDAGISel(TargetMachine, OptLevel) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<RISCVSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  void PreprocessISelDAG() override;
  void PostprocessISelDAG() override;

  void Select(SDNode *Node) override;

  bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                    InlineAsm::ConstraintCode ConstraintID,
                                    std::vector<SDValue> &OutOps) override;

  bool SelectAddrFrameIndex(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool SelectAddrRegImm(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool SelectAddrRegImm9(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool SelectAddrRegImmLsb00000(SDValue Addr, SDValue &Base, SDValue &Offset);

  bool SelectAddrRegRegScale(SDValue Addr, unsigned MaxShiftAmount,
                             SDValue &Base, SDValue &Index, SDValue &Scale);

  template <unsigned MaxShift>
  bool SelectAddrRegRegScale(SDValue Addr, SDValue &Base, SDValue &Index,
                             SDValue &Scale) {
    return SelectAddrRegRegScale(Addr, MaxShift, Base, Index, Scale);
  }

  bool SelectAddrRegZextRegScale(SDValue Addr, unsigned MaxShiftAmount,
                                 unsigned Bits, SDValue &Base, SDValue &Index,
                                 SDValue &Scale);

  template <unsigned MaxShift, unsigned Bits>
  bool SelectAddrRegZextRegScale(SDValue Addr, SDValue &Base, SDValue &Index,
                                 SDValue &Scale) {
    return SelectAddrRegZextRegScale(Addr, MaxShift, Bits, Base, Index, Scale);
  }

  bool SelectAddrRegReg(SDValue Addr, SDValue &Base, SDValue &Offset);

  bool tryShrinkShlLogicImm(SDNode *Node);
  bool trySignedBitfieldExtract(SDNode *Node);
  bool trySignedBitfieldInsertInSign(SDNode *Node);
  bool trySignedBitfieldInsertInMask(SDNode *Node);
  bool tryBitfieldInsertOpFromXor(SDNode *Node);
  bool tryBitfieldInsertOpFromOrAndImm(SDNode *Node);
  bool tryUnsignedBitfieldExtract(SDNode *Node, const SDLoc &DL, MVT VT,
                                  SDValue X, unsigned Msb, unsigned Lsb);
  bool tryUnsignedBitfieldInsertInZero(SDNode *Node, const SDLoc &DL, MVT VT,
                                       SDValue X, unsigned Msb, unsigned Lsb);
  bool tryIndexedLoad(SDNode *Node);

  bool selectShiftMask(SDValue N, unsigned ShiftWidth, SDValue &ShAmt);
  bool selectShiftMaskXLen(SDValue N, SDValue &ShAmt) {
    return selectShiftMask(N, Subtarget->getXLen(), ShAmt);
  }
  bool selectShiftMask32(SDValue N, SDValue &ShAmt) {
    return selectShiftMask(N, 32, ShAmt);
  }

  bool selectSETCC(SDValue N, ISD::CondCode ExpectedCCVal, SDValue &Val);
  bool selectSETNE(SDValue N, SDValue &Val) {
    return selectSETCC(N, ISD::SETNE, Val);
  }
  bool selectSETEQ(SDValue N, SDValue &Val) {
    return selectSETCC(N, ISD::SETEQ, Val);
  }

  bool selectSExtBits(SDValue N, unsigned Bits, SDValue &Val);
  template <unsigned Bits> bool selectSExtBits(SDValue N, SDValue &Val) {
    return selectSExtBits(N, Bits, Val);
  }
  bool selectZExtBits(SDValue N, unsigned Bits, SDValue &Val);
  template <unsigned Bits> bool selectZExtBits(SDValue N, SDValue &Val) {
    return selectZExtBits(N, Bits, Val);
  }

  bool selectSHXADDOp(SDValue N, unsigned ShAmt, SDValue &Val);
  template <unsigned ShAmt> bool selectSHXADDOp(SDValue N, SDValue &Val) {
    return selectSHXADDOp(N, ShAmt, Val);
  }

  bool selectSHXADD_UWOp(SDValue N, unsigned ShAmt, SDValue &Val);
  template <unsigned ShAmt> bool selectSHXADD_UWOp(SDValue N, SDValue &Val) {
    return selectSHXADD_UWOp(N, ShAmt, Val);
  }

  bool selectZExtImm32(SDValue N, SDValue &Val);
  bool selectNegImm(SDValue N, SDValue &Val);
  bool selectInvLogicImm(SDValue N, SDValue &Val);

  bool orDisjoint(const SDNode *Node) const;
  bool hasAllNBitUsers(SDNode *Node, unsigned Bits,
                       const unsigned Depth = 0) const;
  bool hasAllBUsers(SDNode *Node) const { return hasAllNBitUsers(Node, 8); }
  bool hasAllHUsers(SDNode *Node) const { return hasAllNBitUsers(Node, 16); }
  bool hasAllWUsers(SDNode *Node) const { return hasAllNBitUsers(Node, 32); }

  bool selectSimm5Shl2(SDValue N, SDValue &Simm5, SDValue &Shl2);

  bool selectVLOp(SDValue N, SDValue &VL);

  bool selectVSplat(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5(SDValue N, SDValue &SplatVal);
  bool selectVSplatUimm(SDValue N, unsigned Bits, SDValue &SplatVal);
  template <unsigned Bits> bool selectVSplatUimmBits(SDValue N, SDValue &Val) {
    return selectVSplatUimm(N, Bits, Val);
  }
  bool selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5Plus1NoDec(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5Plus1NonZero(SDValue N, SDValue &SplatVal);
  bool selectVSplatImm64Neg(SDValue N, SDValue &SplatVal);
  // Matches the splat of a value which can be extended or truncated, such that
  // only the bottom 8 bits are preserved.
  bool selectLow8BitsVSplat(SDValue N, SDValue &SplatVal);
  bool selectScalarFPAsInt(SDValue N, SDValue &Imm);

  bool selectRVVSimm5(SDValue N, unsigned Width, SDValue &Imm);
  template <unsigned Width> bool selectRVVSimm5(SDValue N, SDValue &Imm) {
    return selectRVVSimm5(N, Width, Imm);
  }

  void addVectorLoadStoreOperands(SDNode *Node, unsigned SEWImm,
                                  const SDLoc &DL, unsigned CurOp,
                                  bool IsMasked, bool IsStridedOrIndexed,
                                  SmallVectorImpl<SDValue> &Operands,
                                  bool IsLoad = false, MVT *IndexVT = nullptr);

  void selectVLSEG(SDNode *Node, unsigned NF, bool IsMasked, bool IsStrided);
  void selectVLSEGFF(SDNode *Node, unsigned NF, bool IsMasked);
  void selectVLXSEG(SDNode *Node, unsigned NF, bool IsMasked, bool IsOrdered);
  void selectVSSEG(SDNode *Node, unsigned NF, bool IsMasked, bool IsStrided);
  void selectVSXSEG(SDNode *Node, unsigned NF, bool IsMasked, bool IsOrdered);

  void selectVSETVLI(SDNode *Node);

  void selectSF_VC_X_SE(SDNode *Node);

  // Return the RISC-V condition code that matches the given DAG integer
  // condition code. The CondCode must be one of those supported by the RISC-V
  // ISA (see translateSetCCForBranch).
  static RISCVCC::CondCode getRISCVCCForIntCC(ISD::CondCode CC) {
    switch (CC) {
    default:
      llvm_unreachable("Unsupported CondCode");
    case ISD::SETEQ:
      return RISCVCC::COND_EQ;
    case ISD::SETNE:
      return RISCVCC::COND_NE;
    case ISD::SETLT:
      return RISCVCC::COND_LT;
    case ISD::SETGE:
      return RISCVCC::COND_GE;
    case ISD::SETULT:
      return RISCVCC::COND_LTU;
    case ISD::SETUGE:
      return RISCVCC::COND_GEU;
    }
  }

// Include the pieces autogenerated from the target description.
#define GET_DAGISEL_DECL
#include "RISCVGenDAGISel.inc"

private:
  bool doPeepholeSExtW(SDNode *Node);
  bool doPeepholeMaskedRVV(MachineSDNode *Node);
  bool doPeepholeNoRegPassThru();
  bool performCombineVMergeAndVOps(SDNode *N);
  bool selectImm64IfCheaper(int64_t Imm, int64_t OrigImm, SDValue N,
                            SDValue &Val);
};

class RISCVDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;
  explicit RISCVDAGToDAGISelLegacy(RISCVTargetMachine &TargetMachine,
                                   CodeGenOptLevel OptLevel);
};

} // namespace llvm

#endif
