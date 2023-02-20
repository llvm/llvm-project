//===---- RISCVISelDAGToDAG.h - A dag to dag inst selector for RISCV ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the RISCV target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVISELDAGTODAG_H
#define LLVM_LIB_TARGET_RISCV_RISCVISELDAGTODAG_H

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Support/KnownBits.h"

// RISCV-specific code to select RISCV machine instructions for
// SelectionDAG operations.
namespace llvm {
class RISCVDAGToDAGISel : public SelectionDAGISel {
  const RISCVSubtarget *Subtarget = nullptr;

public:
  static char ID;

  RISCVDAGToDAGISel() = delete;

  explicit RISCVDAGToDAGISel(RISCVTargetMachine &TargetMachine,
                             CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(ID, TargetMachine, OptLevel) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<RISCVSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  void PreprocessISelDAG() override;
  void PostprocessISelDAG() override;

  void Select(SDNode *Node) override;

  bool SelectInlineAsmMemoryOperand(const SDValue &Op, unsigned ConstraintID,
                                    std::vector<SDValue> &OutOps) override;

  bool SelectAddrFrameIndex(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool SelectFrameAddrRegImm(SDValue Addr, SDValue &Base, SDValue &Offset);
  bool SelectAddrRegImm(SDValue Addr, SDValue &Base, SDValue &Offset);

  bool tryShrinkShlLogicImm(SDNode *Node);
  bool trySignedBitfieldExtract(SDNode *Node);

  bool selectShiftMask(SDValue N, unsigned ShiftWidth, SDValue &ShAmt);
  bool selectShiftMaskXLen(SDValue N, SDValue &ShAmt) {
    return selectShiftMask(N, Subtarget->getXLen(), ShAmt);
  }
  bool selectShiftMask32(SDValue N, SDValue &ShAmt) {
    return selectShiftMask(N, 32, ShAmt);
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

  bool hasAllNBitUsers(SDNode *Node, unsigned Bits,
                       const unsigned Depth = 0) const;
  bool hasAllHUsers(SDNode *Node) const { return hasAllNBitUsers(Node, 16); }
  bool hasAllWUsers(SDNode *Node) const { return hasAllNBitUsers(Node, 32); }

  bool selectVLOp(SDValue N, SDValue &VL);

  bool selectVSplat(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5(SDValue N, SDValue &SplatVal);
  bool selectVSplatUimm5(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5Plus1(SDValue N, SDValue &SplatVal);
  bool selectVSplatSimm5Plus1NonZero(SDValue N, SDValue &SplatVal);
  bool selectFPImm(SDValue N, SDValue &Imm);

  bool selectRVVSimm5(SDValue N, unsigned Width, SDValue &Imm);
  template <unsigned Width> bool selectRVVSimm5(SDValue N, SDValue &Imm) {
    return selectRVVSimm5(N, Width, Imm);
  }

  void addVectorLoadStoreOperands(SDNode *Node, unsigned SEWImm,
                                  const SDLoc &DL, unsigned CurOp,
                                  bool IsMasked, bool IsStridedOrIndexed,
                                  SmallVectorImpl<SDValue> &Operands,
                                  bool IsLoad = false, MVT *IndexVT = nullptr);

  void selectVLSEG(SDNode *Node, bool IsMasked, bool IsStrided);
  void selectVLSEGFF(SDNode *Node, bool IsMasked);
  void selectVLXSEG(SDNode *Node, bool IsMasked, bool IsOrdered);
  void selectVSSEG(SDNode *Node, bool IsMasked, bool IsStrided);
  void selectVSXSEG(SDNode *Node, bool IsMasked, bool IsOrdered);

  void selectVSETVLI(SDNode *Node);

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
#include "RISCVGenDAGISel.inc"

private:
  bool doPeepholeSExtW(SDNode *Node);
  bool doPeepholeMaskedRVV(SDNode *Node);
  bool doPeepholeMergeVVMFold();
  bool performVMergeToVAdd(SDNode *N);
  bool performCombineVMergeAndVOps(SDNode *N, bool IsTA);
};

namespace RISCV {
struct VLSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t IsTU : 1;
  uint16_t Strided : 1;
  uint16_t FF : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VLXSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t IsTU : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct VSSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VSXSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct VLEPseudo {
  uint16_t Masked : 1;
  uint16_t IsTU : 1;
  uint16_t Strided : 1;
  uint16_t FF : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VSEPseudo {
  uint16_t Masked :1;
  uint16_t Strided : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VLX_VSXPseudo {
  uint16_t Masked : 1;
  uint16_t IsTU : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct RISCVMaskedPseudoInfo {
  uint16_t MaskedPseudo;
  uint16_t UnmaskedPseudo;
  uint16_t UnmaskedTUPseudo;
  uint8_t MaskOpIdx;
};

#define GET_RISCVVSSEGTable_DECL
#define GET_RISCVVLSEGTable_DECL
#define GET_RISCVVLXSEGTable_DECL
#define GET_RISCVVSXSEGTable_DECL
#define GET_RISCVVLETable_DECL
#define GET_RISCVVSETable_DECL
#define GET_RISCVVLXTable_DECL
#define GET_RISCVVSXTable_DECL
#define GET_RISCVMaskedPseudosTable_DECL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCV

} // namespace llvm

#endif
