//===-- AArch64ISelDAGToDAG.cpp - A dag to dag inst selector for AArch64 --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the AArch64 target.
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetMachine.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Function.h" // To access function attributes.
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-isel"

//===--------------------------------------------------------------------===//
/// AArch64DAGToDAGISel - AArch64 specific code to select AArch64 machine
/// instructions for SelectionDAG operations.
///
namespace {

class AArch64DAGToDAGISel : public SelectionDAGISel {

  /// Subtarget - Keep a pointer to the AArch64Subtarget around so that we can
  /// make the right decision when generating code for different targets.
  const AArch64Subtarget *Subtarget;

  bool ForCodeSize;

public:
  explicit AArch64DAGToDAGISel(AArch64TargetMachine &tm,
                               CodeGenOpt::Level OptLevel)
      : SelectionDAGISel(tm, OptLevel), Subtarget(nullptr),
        ForCodeSize(false) {}

  const char *getPassName() const override {
    return "AArch64 Instruction Selection";
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    ForCodeSize = MF.getFunction()->optForSize();
    Subtarget = &MF.getSubtarget<AArch64Subtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  SDNode *Select(SDNode *Node) override;

  /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
  /// inline asm expressions.
  bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                    unsigned ConstraintID,
                                    std::vector<SDValue> &OutOps) override;

  SDNode *SelectMLAV64LaneV128(SDNode *N);
  SDNode *SelectMULLV64LaneV128(unsigned IntNo, SDNode *N);
  bool SelectArithExtendedRegister(SDValue N, SDValue &Reg, SDValue &Shift);
  bool SelectArithImmed(SDValue N, SDValue &Val, SDValue &Shift);
  bool SelectNegArithImmed(SDValue N, SDValue &Val, SDValue &Shift);
  bool SelectArithShiftedRegister(SDValue N, SDValue &Reg, SDValue &Shift) {
    return SelectShiftedRegister(N, false, Reg, Shift);
  }
  bool SelectLogicalShiftedRegister(SDValue N, SDValue &Reg, SDValue &Shift) {
    return SelectShiftedRegister(N, true, Reg, Shift);
  }
  bool SelectAddrModeIndexed7S8(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed7S(N, 1, Base, OffImm);
  }
  bool SelectAddrModeIndexed7S16(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed7S(N, 2, Base, OffImm);
  }
  bool SelectAddrModeIndexed7S32(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed7S(N, 4, Base, OffImm);
  }
  bool SelectAddrModeIndexed7S64(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed7S(N, 8, Base, OffImm);
  }
  bool SelectAddrModeIndexed7S128(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed7S(N, 16, Base, OffImm);
  }
  bool SelectAddrModeIndexed8(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 1, Base, OffImm);
  }
  bool SelectAddrModeIndexed16(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 2, Base, OffImm);
  }
  bool SelectAddrModeIndexed32(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 4, Base, OffImm);
  }
  bool SelectAddrModeIndexed64(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 8, Base, OffImm);
  }
  bool SelectAddrModeIndexed128(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeIndexed(N, 16, Base, OffImm);
  }
  bool SelectAddrModeUnscaled8(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 1, Base, OffImm);
  }
  bool SelectAddrModeUnscaled16(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 2, Base, OffImm);
  }
  bool SelectAddrModeUnscaled32(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 4, Base, OffImm);
  }
  bool SelectAddrModeUnscaled64(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 8, Base, OffImm);
  }
  bool SelectAddrModeUnscaled128(SDValue N, SDValue &Base, SDValue &OffImm) {
    return SelectAddrModeUnscaled(N, 16, Base, OffImm);
  }

  template<int Width>
  bool SelectAddrModeWRO(SDValue N, SDValue &Base, SDValue &Offset,
                         SDValue &SignExtend, SDValue &DoShift) {
    return SelectAddrModeWRO(N, Width / 8, Base, Offset, SignExtend, DoShift);
  }

  template<int Width>
  bool SelectAddrModeXRO(SDValue N, SDValue &Base, SDValue &Offset,
                         SDValue &SignExtend, SDValue &DoShift) {
    return SelectAddrModeXRO(N, Width / 8, Base, Offset, SignExtend, DoShift);
  }


  /// Form sequences of consecutive 64/128-bit registers for use in NEON
  /// instructions making use of a vector-list (e.g. ldN, tbl). Vecs must have
  /// between 1 and 4 elements. If it contains a single element that is returned
  /// unchanged; otherwise a REG_SEQUENCE value is returned.
  SDValue createDTuple(ArrayRef<SDValue> Vecs);
  SDValue createQTuple(ArrayRef<SDValue> Vecs);

  /// Generic helper for the createDTuple/createQTuple
  /// functions. Those should almost always be called instead.
  SDValue createTuple(ArrayRef<SDValue> Vecs, const unsigned RegClassIDs[],
                      const unsigned SubRegs[]);

  SDNode *SelectTable(SDNode *N, unsigned NumVecs, unsigned Opc, bool isExt);

  SDNode *SelectIndexedLoad(SDNode *N, bool &Done);

  SDNode *SelectLoad(SDNode *N, unsigned NumVecs, unsigned Opc,
                     unsigned SubRegIdx);
  SDNode *SelectPostLoad(SDNode *N, unsigned NumVecs, unsigned Opc,
                         unsigned SubRegIdx);
  SDNode *SelectLoadLane(SDNode *N, unsigned NumVecs, unsigned Opc);
  SDNode *SelectPostLoadLane(SDNode *N, unsigned NumVecs, unsigned Opc);

  SDNode *SelectStore(SDNode *N, unsigned NumVecs, unsigned Opc);
  SDNode *SelectPostStore(SDNode *N, unsigned NumVecs, unsigned Opc);
  SDNode *SelectStoreLane(SDNode *N, unsigned NumVecs, unsigned Opc);
  SDNode *SelectPostStoreLane(SDNode *N, unsigned NumVecs, unsigned Opc);

  SDNode *SelectBitfieldExtractOp(SDNode *N);
  SDNode *SelectBitfieldInsertOp(SDNode *N);
  SDNode *SelectBitfieldInsertInZeroOp(SDNode *N);

  SDNode *SelectReadRegister(SDNode *N);
  SDNode *SelectWriteRegister(SDNode *N);

// Include the pieces autogenerated from the target description.
#include "AArch64GenDAGISel.inc"

private:
  bool SelectShiftedRegister(SDValue N, bool AllowROR, SDValue &Reg,
                             SDValue &Shift);
  bool SelectAddrModeIndexed7S(SDValue N, unsigned Size, SDValue &Base,
                               SDValue &OffImm);
  bool SelectAddrModeIndexed(SDValue N, unsigned Size, SDValue &Base,
                             SDValue &OffImm);
  bool SelectAddrModeUnscaled(SDValue N, unsigned Size, SDValue &Base,
                              SDValue &OffImm);
  bool SelectAddrModeWRO(SDValue N, unsigned Size, SDValue &Base,
                         SDValue &Offset, SDValue &SignExtend,
                         SDValue &DoShift);
  bool SelectAddrModeXRO(SDValue N, unsigned Size, SDValue &Base,
                         SDValue &Offset, SDValue &SignExtend,
                         SDValue &DoShift);
  bool isWorthFolding(SDValue V) const;
  bool SelectExtendedSHL(SDValue N, unsigned Size, bool WantExtend,
                         SDValue &Offset, SDValue &SignExtend);

  template<unsigned RegWidth>
  bool SelectCVTFixedPosOperand(SDValue N, SDValue &FixedPos) {
    return SelectCVTFixedPosOperand(N, FixedPos, RegWidth);
  }

  bool SelectCVTFixedPosOperand(SDValue N, SDValue &FixedPos, unsigned Width);
};
} // end anonymous namespace

/// isIntImmediate - This method tests to see if the node is a constant
/// operand. If so Imm will receive the 32-bit value.
static bool isIntImmediate(const SDNode *N, uint64_t &Imm) {
  if (const ConstantSDNode *C = dyn_cast<const ConstantSDNode>(N)) {
    Imm = C->getZExtValue();
    return true;
  }
  return false;
}

// isIntImmediate - This method tests to see if a constant operand.
// If so Imm will receive the value.
static bool isIntImmediate(SDValue N, uint64_t &Imm) {
  return isIntImmediate(N.getNode(), Imm);
}

// isOpcWithIntImmediate - This method tests to see if the node is a specific
// opcode and that it has a immediate integer right operand.
// If so Imm will receive the 32 bit value.
static bool isOpcWithIntImmediate(const SDNode *N, unsigned Opc,
                                  uint64_t &Imm) {
  return N->getOpcode() == Opc &&
         isIntImmediate(N->getOperand(1).getNode(), Imm);
}

bool AArch64DAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, unsigned ConstraintID, std::vector<SDValue> &OutOps) {
  switch(ConstraintID) {
  default:
    llvm_unreachable("Unexpected asm memory constraint");
  case InlineAsm::Constraint_i:
  case InlineAsm::Constraint_m:
  case InlineAsm::Constraint_Q:
    // Require the address to be in a register.  That is safe for all AArch64
    // variants and it is hard to do anything much smarter without knowing
    // how the operand is used.
    OutOps.push_back(Op);
    return false;
  }
  return true;
}

/// SelectArithImmed - Select an immediate value that can be represented as
/// a 12-bit value shifted left by either 0 or 12.  If so, return true with
/// Val set to the 12-bit value and Shift set to the shifter operand.
bool AArch64DAGToDAGISel::SelectArithImmed(SDValue N, SDValue &Val,
                                           SDValue &Shift) {
  // This function is called from the addsub_shifted_imm ComplexPattern,
  // which lists [imm] as the list of opcode it's interested in, however
  // we still need to check whether the operand is actually an immediate
  // here because the ComplexPattern opcode list is only used in
  // root-level opcode matching.
  if (!isa<ConstantSDNode>(N.getNode()))
    return false;

  uint64_t Immed = cast<ConstantSDNode>(N.getNode())->getZExtValue();
  unsigned ShiftAmt;

  if (Immed >> 12 == 0) {
    ShiftAmt = 0;
  } else if ((Immed & 0xfff) == 0 && Immed >> 24 == 0) {
    ShiftAmt = 12;
    Immed = Immed >> 12;
  } else
    return false;

  unsigned ShVal = AArch64_AM::getShifterImm(AArch64_AM::LSL, ShiftAmt);
  SDLoc dl(N);
  Val = CurDAG->getTargetConstant(Immed, dl, MVT::i32);
  Shift = CurDAG->getTargetConstant(ShVal, dl, MVT::i32);
  return true;
}

/// SelectNegArithImmed - As above, but negates the value before trying to
/// select it.
bool AArch64DAGToDAGISel::SelectNegArithImmed(SDValue N, SDValue &Val,
                                              SDValue &Shift) {
  // This function is called from the addsub_shifted_imm ComplexPattern,
  // which lists [imm] as the list of opcode it's interested in, however
  // we still need to check whether the operand is actually an immediate
  // here because the ComplexPattern opcode list is only used in
  // root-level opcode matching.
  if (!isa<ConstantSDNode>(N.getNode()))
    return false;

  // The immediate operand must be a 24-bit zero-extended immediate.
  uint64_t Immed = cast<ConstantSDNode>(N.getNode())->getZExtValue();

  // This negation is almost always valid, but "cmp wN, #0" and "cmn wN, #0"
  // have the opposite effect on the C flag, so this pattern mustn't match under
  // those circumstances.
  if (Immed == 0)
    return false;

  if (N.getValueType() == MVT::i32)
    Immed = ~((uint32_t)Immed) + 1;
  else
    Immed = ~Immed + 1ULL;
  if (Immed & 0xFFFFFFFFFF000000ULL)
    return false;

  Immed &= 0xFFFFFFULL;
  return SelectArithImmed(CurDAG->getConstant(Immed, SDLoc(N), MVT::i32), Val,
                          Shift);
}

/// getShiftTypeForNode - Translate a shift node to the corresponding
/// ShiftType value.
static AArch64_AM::ShiftExtendType getShiftTypeForNode(SDValue N) {
  switch (N.getOpcode()) {
  default:
    return AArch64_AM::InvalidShiftExtend;
  case ISD::SHL:
    return AArch64_AM::LSL;
  case ISD::SRL:
    return AArch64_AM::LSR;
  case ISD::SRA:
    return AArch64_AM::ASR;
  case ISD::ROTR:
    return AArch64_AM::ROR;
  }
}

/// \brief Determine whether it is worth to fold V into an extended register.
bool AArch64DAGToDAGISel::isWorthFolding(SDValue V) const {
  // it hurts if the value is used at least twice, unless we are optimizing
  // for code size.
  if (ForCodeSize || V.hasOneUse())
    return true;
  return false;
}

/// SelectShiftedRegister - Select a "shifted register" operand.  If the value
/// is not shifted, set the Shift operand to default of "LSL 0".  The logical
/// instructions allow the shifted register to be rotated, but the arithmetic
/// instructions do not.  The AllowROR parameter specifies whether ROR is
/// supported.
bool AArch64DAGToDAGISel::SelectShiftedRegister(SDValue N, bool AllowROR,
                                                SDValue &Reg, SDValue &Shift) {
  AArch64_AM::ShiftExtendType ShType = getShiftTypeForNode(N);
  if (ShType == AArch64_AM::InvalidShiftExtend)
    return false;
  if (!AllowROR && ShType == AArch64_AM::ROR)
    return false;

  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    unsigned BitSize = N.getValueType().getSizeInBits();
    unsigned Val = RHS->getZExtValue() & (BitSize - 1);
    unsigned ShVal = AArch64_AM::getShifterImm(ShType, Val);

    Reg = N.getOperand(0);
    Shift = CurDAG->getTargetConstant(ShVal, SDLoc(N), MVT::i32);
    return isWorthFolding(N);
  }

  return false;
}

/// getExtendTypeForNode - Translate an extend node to the corresponding
/// ExtendType value.
static AArch64_AM::ShiftExtendType
getExtendTypeForNode(SDValue N, bool IsLoadStore = false) {
  if (N.getOpcode() == ISD::SIGN_EXTEND ||
      N.getOpcode() == ISD::SIGN_EXTEND_INREG) {
    EVT SrcVT;
    if (N.getOpcode() == ISD::SIGN_EXTEND_INREG)
      SrcVT = cast<VTSDNode>(N.getOperand(1))->getVT();
    else
      SrcVT = N.getOperand(0).getValueType();

    if (!IsLoadStore && SrcVT == MVT::i8)
      return AArch64_AM::SXTB;
    else if (!IsLoadStore && SrcVT == MVT::i16)
      return AArch64_AM::SXTH;
    else if (SrcVT == MVT::i32)
      return AArch64_AM::SXTW;
    assert(SrcVT != MVT::i64 && "extend from 64-bits?");

    return AArch64_AM::InvalidShiftExtend;
  } else if (N.getOpcode() == ISD::ZERO_EXTEND ||
             N.getOpcode() == ISD::ANY_EXTEND) {
    EVT SrcVT = N.getOperand(0).getValueType();
    if (!IsLoadStore && SrcVT == MVT::i8)
      return AArch64_AM::UXTB;
    else if (!IsLoadStore && SrcVT == MVT::i16)
      return AArch64_AM::UXTH;
    else if (SrcVT == MVT::i32)
      return AArch64_AM::UXTW;
    assert(SrcVT != MVT::i64 && "extend from 64-bits?");

    return AArch64_AM::InvalidShiftExtend;
  } else if (N.getOpcode() == ISD::AND) {
    ConstantSDNode *CSD = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!CSD)
      return AArch64_AM::InvalidShiftExtend;
    uint64_t AndMask = CSD->getZExtValue();

    switch (AndMask) {
    default:
      return AArch64_AM::InvalidShiftExtend;
    case 0xFF:
      return !IsLoadStore ? AArch64_AM::UXTB : AArch64_AM::InvalidShiftExtend;
    case 0xFFFF:
      return !IsLoadStore ? AArch64_AM::UXTH : AArch64_AM::InvalidShiftExtend;
    case 0xFFFFFFFF:
      return AArch64_AM::UXTW;
    }
  }

  return AArch64_AM::InvalidShiftExtend;
}

// Helper for SelectMLAV64LaneV128 - Recognize high lane extracts.
static bool checkHighLaneIndex(SDNode *DL, SDValue &LaneOp, int &LaneIdx) {
  if (DL->getOpcode() != AArch64ISD::DUPLANE16 &&
      DL->getOpcode() != AArch64ISD::DUPLANE32)
    return false;

  SDValue SV = DL->getOperand(0);
  if (SV.getOpcode() != ISD::INSERT_SUBVECTOR)
    return false;

  SDValue EV = SV.getOperand(1);
  if (EV.getOpcode() != ISD::EXTRACT_SUBVECTOR)
    return false;

  ConstantSDNode *DLidx = cast<ConstantSDNode>(DL->getOperand(1).getNode());
  ConstantSDNode *EVidx = cast<ConstantSDNode>(EV.getOperand(1).getNode());
  LaneIdx = DLidx->getSExtValue() + EVidx->getSExtValue();
  LaneOp = EV.getOperand(0);

  return true;
}

// Helper for SelectOpcV64LaneV128 - Recognize operations where one operand is a
// high lane extract.
static bool checkV64LaneV128(SDValue Op0, SDValue Op1, SDValue &StdOp,
                             SDValue &LaneOp, int &LaneIdx) {

  if (!checkHighLaneIndex(Op0.getNode(), LaneOp, LaneIdx)) {
    std::swap(Op0, Op1);
    if (!checkHighLaneIndex(Op0.getNode(), LaneOp, LaneIdx))
      return false;
  }
  StdOp = Op1;
  return true;
}

/// SelectMLAV64LaneV128 - AArch64 supports vector MLAs where one multiplicand
/// is a lane in the upper half of a 128-bit vector.  Recognize and select this
/// so that we don't emit unnecessary lane extracts.
SDNode *AArch64DAGToDAGISel::SelectMLAV64LaneV128(SDNode *N) {
  SDLoc dl(N);
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  SDValue MLAOp1;   // Will hold ordinary multiplicand for MLA.
  SDValue MLAOp2;   // Will hold lane-accessed multiplicand for MLA.
  int LaneIdx = -1; // Will hold the lane index.

  if (Op1.getOpcode() != ISD::MUL ||
      !checkV64LaneV128(Op1.getOperand(0), Op1.getOperand(1), MLAOp1, MLAOp2,
                        LaneIdx)) {
    std::swap(Op0, Op1);
    if (Op1.getOpcode() != ISD::MUL ||
        !checkV64LaneV128(Op1.getOperand(0), Op1.getOperand(1), MLAOp1, MLAOp2,
                          LaneIdx))
      return nullptr;
  }

  SDValue LaneIdxVal = CurDAG->getTargetConstant(LaneIdx, dl, MVT::i64);

  SDValue Ops[] = { Op0, MLAOp1, MLAOp2, LaneIdxVal };

  unsigned MLAOpc = ~0U;

  switch (N->getSimpleValueType(0).SimpleTy) {
  default:
    llvm_unreachable("Unrecognized MLA.");
  case MVT::v4i16:
    MLAOpc = AArch64::MLAv4i16_indexed;
    break;
  case MVT::v8i16:
    MLAOpc = AArch64::MLAv8i16_indexed;
    break;
  case MVT::v2i32:
    MLAOpc = AArch64::MLAv2i32_indexed;
    break;
  case MVT::v4i32:
    MLAOpc = AArch64::MLAv4i32_indexed;
    break;
  }

  return CurDAG->getMachineNode(MLAOpc, dl, N->getValueType(0), Ops);
}

SDNode *AArch64DAGToDAGISel::SelectMULLV64LaneV128(unsigned IntNo, SDNode *N) {
  SDLoc dl(N);
  SDValue SMULLOp0;
  SDValue SMULLOp1;
  int LaneIdx;

  if (!checkV64LaneV128(N->getOperand(1), N->getOperand(2), SMULLOp0, SMULLOp1,
                        LaneIdx))
    return nullptr;

  SDValue LaneIdxVal = CurDAG->getTargetConstant(LaneIdx, dl, MVT::i64);

  SDValue Ops[] = { SMULLOp0, SMULLOp1, LaneIdxVal };

  unsigned SMULLOpc = ~0U;

  if (IntNo == Intrinsic::aarch64_neon_smull) {
    switch (N->getSimpleValueType(0).SimpleTy) {
    default:
      llvm_unreachable("Unrecognized SMULL.");
    case MVT::v4i32:
      SMULLOpc = AArch64::SMULLv4i16_indexed;
      break;
    case MVT::v2i64:
      SMULLOpc = AArch64::SMULLv2i32_indexed;
      break;
    }
  } else if (IntNo == Intrinsic::aarch64_neon_umull) {
    switch (N->getSimpleValueType(0).SimpleTy) {
    default:
      llvm_unreachable("Unrecognized SMULL.");
    case MVT::v4i32:
      SMULLOpc = AArch64::UMULLv4i16_indexed;
      break;
    case MVT::v2i64:
      SMULLOpc = AArch64::UMULLv2i32_indexed;
      break;
    }
  } else
    llvm_unreachable("Unrecognized intrinsic.");

  return CurDAG->getMachineNode(SMULLOpc, dl, N->getValueType(0), Ops);
}

/// Instructions that accept extend modifiers like UXTW expect the register
/// being extended to be a GPR32, but the incoming DAG might be acting on a
/// GPR64 (either via SEXT_INREG or AND). Extract the appropriate low bits if
/// this is the case.
static SDValue narrowIfNeeded(SelectionDAG *CurDAG, SDValue N) {
  if (N.getValueType() == MVT::i32)
    return N;

  SDLoc dl(N);
  SDValue SubReg = CurDAG->getTargetConstant(AArch64::sub_32, dl, MVT::i32);
  MachineSDNode *Node = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                               dl, MVT::i32, N, SubReg);
  return SDValue(Node, 0);
}


/// SelectArithExtendedRegister - Select a "extended register" operand.  This
/// operand folds in an extend followed by an optional left shift.
bool AArch64DAGToDAGISel::SelectArithExtendedRegister(SDValue N, SDValue &Reg,
                                                      SDValue &Shift) {
  unsigned ShiftVal = 0;
  AArch64_AM::ShiftExtendType Ext;

  if (N.getOpcode() == ISD::SHL) {
    ConstantSDNode *CSD = dyn_cast<ConstantSDNode>(N.getOperand(1));
    if (!CSD)
      return false;
    ShiftVal = CSD->getZExtValue();
    if (ShiftVal > 4)
      return false;

    Ext = getExtendTypeForNode(N.getOperand(0));
    if (Ext == AArch64_AM::InvalidShiftExtend)
      return false;

    Reg = N.getOperand(0).getOperand(0);
  } else {
    Ext = getExtendTypeForNode(N);
    if (Ext == AArch64_AM::InvalidShiftExtend)
      return false;

    Reg = N.getOperand(0);
  }

  // AArch64 mandates that the RHS of the operation must use the smallest
  // register class that could contain the size being extended from.  Thus,
  // if we're folding a (sext i8), we need the RHS to be a GPR32, even though
  // there might not be an actual 32-bit value in the program.  We can
  // (harmlessly) synthesize one by injected an EXTRACT_SUBREG here.
  assert(Ext != AArch64_AM::UXTX && Ext != AArch64_AM::SXTX);
  Reg = narrowIfNeeded(CurDAG, Reg);
  Shift = CurDAG->getTargetConstant(getArithExtendImm(Ext, ShiftVal), SDLoc(N),
                                    MVT::i32);
  return isWorthFolding(N);
}

/// If there's a use of this ADDlow that's not itself a load/store then we'll
/// need to create a real ADD instruction from it anyway and there's no point in
/// folding it into the mem op. Theoretically, it shouldn't matter, but there's
/// a single pseudo-instruction for an ADRP/ADD pair so over-aggressive folding
/// leads to duplicated ADRP instructions.
static bool isWorthFoldingADDlow(SDValue N) {
  for (auto Use : N->uses()) {
    if (Use->getOpcode() != ISD::LOAD && Use->getOpcode() != ISD::STORE &&
        Use->getOpcode() != ISD::ATOMIC_LOAD &&
        Use->getOpcode() != ISD::ATOMIC_STORE)
      return false;

    // ldar and stlr have much more restrictive addressing modes (just a
    // register).
    if (cast<MemSDNode>(Use)->getOrdering() > Monotonic)
      return false;
  }

  return true;
}

/// SelectAddrModeIndexed7S - Select a "register plus scaled signed 7-bit
/// immediate" address.  The "Size" argument is the size in bytes of the memory
/// reference, which determines the scale.
bool AArch64DAGToDAGISel::SelectAddrModeIndexed7S(SDValue N, unsigned Size,
                                                  SDValue &Base,
                                                  SDValue &OffImm) {
  SDLoc dl(N);
  const DataLayout &DL = CurDAG->getDataLayout();
  const TargetLowering *TLI = getTargetLowering();
  if (N.getOpcode() == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    Base = CurDAG->getTargetFrameIndex(FI, TLI->getPointerTy(DL));
    OffImm = CurDAG->getTargetConstant(0, dl, MVT::i64);
    return true;
  }

  // As opposed to the (12-bit) Indexed addressing mode below, the 7-bit signed
  // selected here doesn't support labels/immediates, only base+offset.

  if (CurDAG->isBaseWithConstantOffset(N)) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      int64_t RHSC = RHS->getSExtValue();
      unsigned Scale = Log2_32(Size);
      if ((RHSC & (Size - 1)) == 0 && RHSC >= -(0x40 << Scale) &&
          RHSC < (0x40 << Scale)) {
        Base = N.getOperand(0);
        if (Base.getOpcode() == ISD::FrameIndex) {
          int FI = cast<FrameIndexSDNode>(Base)->getIndex();
          Base = CurDAG->getTargetFrameIndex(FI, TLI->getPointerTy(DL));
        }
        OffImm = CurDAG->getTargetConstant(RHSC >> Scale, dl, MVT::i64);
        return true;
      }
    }
  }

  // Base only. The address will be materialized into a register before
  // the memory is accessed.
  //    add x0, Xbase, #offset
  //    stp x1, x2, [x0]
  Base = N;
  OffImm = CurDAG->getTargetConstant(0, dl, MVT::i64);
  return true;
}

/// SelectAddrModeIndexed - Select a "register plus scaled unsigned 12-bit
/// immediate" address.  The "Size" argument is the size in bytes of the memory
/// reference, which determines the scale.
bool AArch64DAGToDAGISel::SelectAddrModeIndexed(SDValue N, unsigned Size,
                                              SDValue &Base, SDValue &OffImm) {
  SDLoc dl(N);
  const DataLayout &DL = CurDAG->getDataLayout();
  const TargetLowering *TLI = getTargetLowering();
  if (N.getOpcode() == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    Base = CurDAG->getTargetFrameIndex(FI, TLI->getPointerTy(DL));
    OffImm = CurDAG->getTargetConstant(0, dl, MVT::i64);
    return true;
  }

  if (N.getOpcode() == AArch64ISD::ADDlow && isWorthFoldingADDlow(N)) {
    GlobalAddressSDNode *GAN =
        dyn_cast<GlobalAddressSDNode>(N.getOperand(1).getNode());
    Base = N.getOperand(0);
    OffImm = N.getOperand(1);
    if (!GAN)
      return true;

    const GlobalValue *GV = GAN->getGlobal();
    unsigned Alignment = GV->getAlignment();
    Type *Ty = GV->getType()->getElementType();
    if (Alignment == 0 && Ty->isSized())
      Alignment = DL.getABITypeAlignment(Ty);

    if (Alignment >= Size)
      return true;
  }

  if (CurDAG->isBaseWithConstantOffset(N)) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      int64_t RHSC = (int64_t)RHS->getZExtValue();
      unsigned Scale = Log2_32(Size);
      if ((RHSC & (Size - 1)) == 0 && RHSC >= 0 && RHSC < (0x1000 << Scale)) {
        Base = N.getOperand(0);
        if (Base.getOpcode() == ISD::FrameIndex) {
          int FI = cast<FrameIndexSDNode>(Base)->getIndex();
          Base = CurDAG->getTargetFrameIndex(FI, TLI->getPointerTy(DL));
        }
        OffImm = CurDAG->getTargetConstant(RHSC >> Scale, dl, MVT::i64);
        return true;
      }
    }
  }

  // Before falling back to our general case, check if the unscaled
  // instructions can handle this. If so, that's preferable.
  if (SelectAddrModeUnscaled(N, Size, Base, OffImm))
    return false;

  // Base only. The address will be materialized into a register before
  // the memory is accessed.
  //    add x0, Xbase, #offset
  //    ldr x0, [x0]
  Base = N;
  OffImm = CurDAG->getTargetConstant(0, dl, MVT::i64);
  return true;
}

/// SelectAddrModeUnscaled - Select a "register plus unscaled signed 9-bit
/// immediate" address.  This should only match when there is an offset that
/// is not valid for a scaled immediate addressing mode.  The "Size" argument
/// is the size in bytes of the memory reference, which is needed here to know
/// what is valid for a scaled immediate.
bool AArch64DAGToDAGISel::SelectAddrModeUnscaled(SDValue N, unsigned Size,
                                                 SDValue &Base,
                                                 SDValue &OffImm) {
  if (!CurDAG->isBaseWithConstantOffset(N))
    return false;
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int64_t RHSC = RHS->getSExtValue();
    // If the offset is valid as a scaled immediate, don't match here.
    if ((RHSC & (Size - 1)) == 0 && RHSC >= 0 &&
        RHSC < (0x1000 << Log2_32(Size)))
      return false;
    if (RHSC >= -256 && RHSC < 256) {
      Base = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        const TargetLowering *TLI = getTargetLowering();
        Base = CurDAG->getTargetFrameIndex(
            FI, TLI->getPointerTy(CurDAG->getDataLayout()));
      }
      OffImm = CurDAG->getTargetConstant(RHSC, SDLoc(N), MVT::i64);
      return true;
    }
  }
  return false;
}

static SDValue Widen(SelectionDAG *CurDAG, SDValue N) {
  SDLoc dl(N);
  SDValue SubReg = CurDAG->getTargetConstant(AArch64::sub_32, dl, MVT::i32);
  SDValue ImpDef = SDValue(
      CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, dl, MVT::i64), 0);
  MachineSDNode *Node = CurDAG->getMachineNode(
      TargetOpcode::INSERT_SUBREG, dl, MVT::i64, ImpDef, N, SubReg);
  return SDValue(Node, 0);
}

/// \brief Check if the given SHL node (\p N), can be used to form an
/// extended register for an addressing mode.
bool AArch64DAGToDAGISel::SelectExtendedSHL(SDValue N, unsigned Size,
                                            bool WantExtend, SDValue &Offset,
                                            SDValue &SignExtend) {
  assert(N.getOpcode() == ISD::SHL && "Invalid opcode.");
  ConstantSDNode *CSD = dyn_cast<ConstantSDNode>(N.getOperand(1));
  if (!CSD || (CSD->getZExtValue() & 0x7) != CSD->getZExtValue())
    return false;

  SDLoc dl(N);
  if (WantExtend) {
    AArch64_AM::ShiftExtendType Ext =
        getExtendTypeForNode(N.getOperand(0), true);
    if (Ext == AArch64_AM::InvalidShiftExtend)
      return false;

    Offset = narrowIfNeeded(CurDAG, N.getOperand(0).getOperand(0));
    SignExtend = CurDAG->getTargetConstant(Ext == AArch64_AM::SXTW, dl,
                                           MVT::i32);
  } else {
    Offset = N.getOperand(0);
    SignExtend = CurDAG->getTargetConstant(0, dl, MVT::i32);
  }

  unsigned LegalShiftVal = Log2_32(Size);
  unsigned ShiftVal = CSD->getZExtValue();

  if (ShiftVal != 0 && ShiftVal != LegalShiftVal)
    return false;

  if (isWorthFolding(N))
    return true;

  return false;
}

bool AArch64DAGToDAGISel::SelectAddrModeWRO(SDValue N, unsigned Size,
                                            SDValue &Base, SDValue &Offset,
                                            SDValue &SignExtend,
                                            SDValue &DoShift) {
  if (N.getOpcode() != ISD::ADD)
    return false;
  SDValue LHS = N.getOperand(0);
  SDValue RHS = N.getOperand(1);
  SDLoc dl(N);

  // We don't want to match immediate adds here, because they are better lowered
  // to the register-immediate addressing modes.
  if (isa<ConstantSDNode>(LHS) || isa<ConstantSDNode>(RHS))
    return false;

  // Check if this particular node is reused in any non-memory related
  // operation.  If yes, do not try to fold this node into the address
  // computation, since the computation will be kept.
  const SDNode *Node = N.getNode();
  for (SDNode *UI : Node->uses()) {
    if (!isa<MemSDNode>(*UI))
      return false;
  }

  // Remember if it is worth folding N when it produces extended register.
  bool IsExtendedRegisterWorthFolding = isWorthFolding(N);

  // Try to match a shifted extend on the RHS.
  if (IsExtendedRegisterWorthFolding && RHS.getOpcode() == ISD::SHL &&
      SelectExtendedSHL(RHS, Size, true, Offset, SignExtend)) {
    Base = LHS;
    DoShift = CurDAG->getTargetConstant(true, dl, MVT::i32);
    return true;
  }

  // Try to match a shifted extend on the LHS.
  if (IsExtendedRegisterWorthFolding && LHS.getOpcode() == ISD::SHL &&
      SelectExtendedSHL(LHS, Size, true, Offset, SignExtend)) {
    Base = RHS;
    DoShift = CurDAG->getTargetConstant(true, dl, MVT::i32);
    return true;
  }

  // There was no shift, whatever else we find.
  DoShift = CurDAG->getTargetConstant(false, dl, MVT::i32);

  AArch64_AM::ShiftExtendType Ext = AArch64_AM::InvalidShiftExtend;
  // Try to match an unshifted extend on the LHS.
  if (IsExtendedRegisterWorthFolding &&
      (Ext = getExtendTypeForNode(LHS, true)) !=
          AArch64_AM::InvalidShiftExtend) {
    Base = RHS;
    Offset = narrowIfNeeded(CurDAG, LHS.getOperand(0));
    SignExtend = CurDAG->getTargetConstant(Ext == AArch64_AM::SXTW, dl,
                                           MVT::i32);
    if (isWorthFolding(LHS))
      return true;
  }

  // Try to match an unshifted extend on the RHS.
  if (IsExtendedRegisterWorthFolding &&
      (Ext = getExtendTypeForNode(RHS, true)) !=
          AArch64_AM::InvalidShiftExtend) {
    Base = LHS;
    Offset = narrowIfNeeded(CurDAG, RHS.getOperand(0));
    SignExtend = CurDAG->getTargetConstant(Ext == AArch64_AM::SXTW, dl,
                                           MVT::i32);
    if (isWorthFolding(RHS))
      return true;
  }

  return false;
}

// Check if the given immediate is preferred by ADD. If an immediate can be
// encoded in an ADD, or it can be encoded in an "ADD LSL #12" and can not be
// encoded by one MOVZ, return true.
static bool isPreferredADD(int64_t ImmOff) {
  // Constant in [0x0, 0xfff] can be encoded in ADD.
  if ((ImmOff & 0xfffffffffffff000LL) == 0x0LL)
    return true;
  // Check if it can be encoded in an "ADD LSL #12".
  if ((ImmOff & 0xffffffffff000fffLL) == 0x0LL)
    // As a single MOVZ is faster than a "ADD of LSL #12", ignore such constant.
    return (ImmOff & 0xffffffffff00ffffLL) != 0x0LL &&
           (ImmOff & 0xffffffffffff0fffLL) != 0x0LL;
  return false;
}

bool AArch64DAGToDAGISel::SelectAddrModeXRO(SDValue N, unsigned Size,
                                            SDValue &Base, SDValue &Offset,
                                            SDValue &SignExtend,
                                            SDValue &DoShift) {
  if (N.getOpcode() != ISD::ADD)
    return false;
  SDValue LHS = N.getOperand(0);
  SDValue RHS = N.getOperand(1);
  SDLoc DL(N);

  // Check if this particular node is reused in any non-memory related
  // operation.  If yes, do not try to fold this node into the address
  // computation, since the computation will be kept.
  const SDNode *Node = N.getNode();
  for (SDNode *UI : Node->uses()) {
    if (!isa<MemSDNode>(*UI))
      return false;
  }

  // Watch out if RHS is a wide immediate, it can not be selected into
  // [BaseReg+Imm] addressing mode. Also it may not be able to be encoded into
  // ADD/SUB. Instead it will use [BaseReg + 0] address mode and generate
  // instructions like:
  //     MOV  X0, WideImmediate
  //     ADD  X1, BaseReg, X0
  //     LDR  X2, [X1, 0]
  // For such situation, using [BaseReg, XReg] addressing mode can save one
  // ADD/SUB:
  //     MOV  X0, WideImmediate
  //     LDR  X2, [BaseReg, X0]
  if (isa<ConstantSDNode>(RHS)) {
    int64_t ImmOff = (int64_t)cast<ConstantSDNode>(RHS)->getZExtValue();
    unsigned Scale = Log2_32(Size);
    // Skip the immediate can be selected by load/store addressing mode.
    // Also skip the immediate can be encoded by a single ADD (SUB is also
    // checked by using -ImmOff).
    if ((ImmOff % Size == 0 && ImmOff >= 0 && ImmOff < (0x1000 << Scale)) ||
        isPreferredADD(ImmOff) || isPreferredADD(-ImmOff))
      return false;

    SDValue Ops[] = { RHS };
    SDNode *MOVI =
        CurDAG->getMachineNode(AArch64::MOVi64imm, DL, MVT::i64, Ops);
    SDValue MOVIV = SDValue(MOVI, 0);
    // This ADD of two X register will be selected into [Reg+Reg] mode.
    N = CurDAG->getNode(ISD::ADD, DL, MVT::i64, LHS, MOVIV);
  }

  // Remember if it is worth folding N when it produces extended register.
  bool IsExtendedRegisterWorthFolding = isWorthFolding(N);

  // Try to match a shifted extend on the RHS.
  if (IsExtendedRegisterWorthFolding && RHS.getOpcode() == ISD::SHL &&
      SelectExtendedSHL(RHS, Size, false, Offset, SignExtend)) {
    Base = LHS;
    DoShift = CurDAG->getTargetConstant(true, DL, MVT::i32);
    return true;
  }

  // Try to match a shifted extend on the LHS.
  if (IsExtendedRegisterWorthFolding && LHS.getOpcode() == ISD::SHL &&
      SelectExtendedSHL(LHS, Size, false, Offset, SignExtend)) {
    Base = RHS;
    DoShift = CurDAG->getTargetConstant(true, DL, MVT::i32);
    return true;
  }

  // Match any non-shifted, non-extend, non-immediate add expression.
  Base = LHS;
  Offset = RHS;
  SignExtend = CurDAG->getTargetConstant(false, DL, MVT::i32);
  DoShift = CurDAG->getTargetConstant(false, DL, MVT::i32);
  // Reg1 + Reg2 is free: no check needed.
  return true;
}

SDValue AArch64DAGToDAGISel::createDTuple(ArrayRef<SDValue> Regs) {
  static const unsigned RegClassIDs[] = {
      AArch64::DDRegClassID, AArch64::DDDRegClassID, AArch64::DDDDRegClassID};
  static const unsigned SubRegs[] = {AArch64::dsub0, AArch64::dsub1,
                                     AArch64::dsub2, AArch64::dsub3};

  return createTuple(Regs, RegClassIDs, SubRegs);
}

SDValue AArch64DAGToDAGISel::createQTuple(ArrayRef<SDValue> Regs) {
  static const unsigned RegClassIDs[] = {
      AArch64::QQRegClassID, AArch64::QQQRegClassID, AArch64::QQQQRegClassID};
  static const unsigned SubRegs[] = {AArch64::qsub0, AArch64::qsub1,
                                     AArch64::qsub2, AArch64::qsub3};

  return createTuple(Regs, RegClassIDs, SubRegs);
}

SDValue AArch64DAGToDAGISel::createTuple(ArrayRef<SDValue> Regs,
                                         const unsigned RegClassIDs[],
                                         const unsigned SubRegs[]) {
  // There's no special register-class for a vector-list of 1 element: it's just
  // a vector.
  if (Regs.size() == 1)
    return Regs[0];

  assert(Regs.size() >= 2 && Regs.size() <= 4);

  SDLoc DL(Regs[0]);

  SmallVector<SDValue, 4> Ops;

  // First operand of REG_SEQUENCE is the desired RegClass.
  Ops.push_back(
      CurDAG->getTargetConstant(RegClassIDs[Regs.size() - 2], DL, MVT::i32));

  // Then we get pairs of source & subregister-position for the components.
  for (unsigned i = 0; i < Regs.size(); ++i) {
    Ops.push_back(Regs[i]);
    Ops.push_back(CurDAG->getTargetConstant(SubRegs[i], DL, MVT::i32));
  }

  SDNode *N =
      CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::Untyped, Ops);
  return SDValue(N, 0);
}

SDNode *AArch64DAGToDAGISel::SelectTable(SDNode *N, unsigned NumVecs,
                                         unsigned Opc, bool isExt) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);

  unsigned ExtOff = isExt;

  // Form a REG_SEQUENCE to force register allocation.
  unsigned Vec0Off = ExtOff + 1;
  SmallVector<SDValue, 4> Regs(N->op_begin() + Vec0Off,
                               N->op_begin() + Vec0Off + NumVecs);
  SDValue RegSeq = createQTuple(Regs);

  SmallVector<SDValue, 6> Ops;
  if (isExt)
    Ops.push_back(N->getOperand(1));
  Ops.push_back(RegSeq);
  Ops.push_back(N->getOperand(NumVecs + ExtOff + 1));
  return CurDAG->getMachineNode(Opc, dl, VT, Ops);
}

SDNode *AArch64DAGToDAGISel::SelectIndexedLoad(SDNode *N, bool &Done) {
  LoadSDNode *LD = cast<LoadSDNode>(N);
  if (LD->isUnindexed())
    return nullptr;
  EVT VT = LD->getMemoryVT();
  EVT DstVT = N->getValueType(0);
  ISD::MemIndexedMode AM = LD->getAddressingMode();
  bool IsPre = AM == ISD::PRE_INC || AM == ISD::PRE_DEC;

  // We're not doing validity checking here. That was done when checking
  // if we should mark the load as indexed or not. We're just selecting
  // the right instruction.
  unsigned Opcode = 0;

  ISD::LoadExtType ExtType = LD->getExtensionType();
  bool InsertTo64 = false;
  if (VT == MVT::i64)
    Opcode = IsPre ? AArch64::LDRXpre : AArch64::LDRXpost;
  else if (VT == MVT::i32) {
    if (ExtType == ISD::NON_EXTLOAD)
      Opcode = IsPre ? AArch64::LDRWpre : AArch64::LDRWpost;
    else if (ExtType == ISD::SEXTLOAD)
      Opcode = IsPre ? AArch64::LDRSWpre : AArch64::LDRSWpost;
    else {
      Opcode = IsPre ? AArch64::LDRWpre : AArch64::LDRWpost;
      InsertTo64 = true;
      // The result of the load is only i32. It's the subreg_to_reg that makes
      // it into an i64.
      DstVT = MVT::i32;
    }
  } else if (VT == MVT::i16) {
    if (ExtType == ISD::SEXTLOAD) {
      if (DstVT == MVT::i64)
        Opcode = IsPre ? AArch64::LDRSHXpre : AArch64::LDRSHXpost;
      else
        Opcode = IsPre ? AArch64::LDRSHWpre : AArch64::LDRSHWpost;
    } else {
      Opcode = IsPre ? AArch64::LDRHHpre : AArch64::LDRHHpost;
      InsertTo64 = DstVT == MVT::i64;
      // The result of the load is only i32. It's the subreg_to_reg that makes
      // it into an i64.
      DstVT = MVT::i32;
    }
  } else if (VT == MVT::i8) {
    if (ExtType == ISD::SEXTLOAD) {
      if (DstVT == MVT::i64)
        Opcode = IsPre ? AArch64::LDRSBXpre : AArch64::LDRSBXpost;
      else
        Opcode = IsPre ? AArch64::LDRSBWpre : AArch64::LDRSBWpost;
    } else {
      Opcode = IsPre ? AArch64::LDRBBpre : AArch64::LDRBBpost;
      InsertTo64 = DstVT == MVT::i64;
      // The result of the load is only i32. It's the subreg_to_reg that makes
      // it into an i64.
      DstVT = MVT::i32;
    }
  } else if (VT == MVT::f16) {
    Opcode = IsPre ? AArch64::LDRHpre : AArch64::LDRHpost;
  } else if (VT == MVT::f32) {
    Opcode = IsPre ? AArch64::LDRSpre : AArch64::LDRSpost;
  } else if (VT == MVT::f64 || VT.is64BitVector()) {
    Opcode = IsPre ? AArch64::LDRDpre : AArch64::LDRDpost;
  } else if (VT.is128BitVector()) {
    Opcode = IsPre ? AArch64::LDRQpre : AArch64::LDRQpost;
  } else
    return nullptr;
  SDValue Chain = LD->getChain();
  SDValue Base = LD->getBasePtr();
  ConstantSDNode *OffsetOp = cast<ConstantSDNode>(LD->getOffset());
  int OffsetVal = (int)OffsetOp->getZExtValue();
  SDLoc dl(N);
  SDValue Offset = CurDAG->getTargetConstant(OffsetVal, dl, MVT::i64);
  SDValue Ops[] = { Base, Offset, Chain };
  SDNode *Res = CurDAG->getMachineNode(Opcode, dl, MVT::i64, DstVT,
                                       MVT::Other, Ops);
  // Either way, we're replacing the node, so tell the caller that.
  Done = true;
  SDValue LoadedVal = SDValue(Res, 1);
  if (InsertTo64) {
    SDValue SubReg = CurDAG->getTargetConstant(AArch64::sub_32, dl, MVT::i32);
    LoadedVal =
        SDValue(CurDAG->getMachineNode(
                    AArch64::SUBREG_TO_REG, dl, MVT::i64,
                    CurDAG->getTargetConstant(0, dl, MVT::i64), LoadedVal,
                    SubReg),
                0);
  }

  ReplaceUses(SDValue(N, 0), LoadedVal);
  ReplaceUses(SDValue(N, 1), SDValue(Res, 0));
  ReplaceUses(SDValue(N, 2), SDValue(Res, 2));

  return nullptr;
}

SDNode *AArch64DAGToDAGISel::SelectLoad(SDNode *N, unsigned NumVecs,
                                        unsigned Opc, unsigned SubRegIdx) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  SDValue Chain = N->getOperand(0);

  SDValue Ops[] = {N->getOperand(2), // Mem operand;
                   Chain};

  const EVT ResTys[] = {MVT::Untyped, MVT::Other};

  SDNode *Ld = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);
  SDValue SuperReg = SDValue(Ld, 0);
  for (unsigned i = 0; i < NumVecs; ++i)
    ReplaceUses(SDValue(N, i),
        CurDAG->getTargetExtractSubreg(SubRegIdx + i, dl, VT, SuperReg));

  ReplaceUses(SDValue(N, NumVecs), SDValue(Ld, 1));
  return nullptr;
}

SDNode *AArch64DAGToDAGISel::SelectPostLoad(SDNode *N, unsigned NumVecs,
                                            unsigned Opc, unsigned SubRegIdx) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  SDValue Chain = N->getOperand(0);

  SDValue Ops[] = {N->getOperand(1), // Mem operand
                   N->getOperand(2), // Incremental
                   Chain};

  const EVT ResTys[] = {MVT::i64, // Type of the write back register
                        MVT::Untyped, MVT::Other};

  SDNode *Ld = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);

  // Update uses of write back register
  ReplaceUses(SDValue(N, NumVecs), SDValue(Ld, 0));

  // Update uses of vector list
  SDValue SuperReg = SDValue(Ld, 1);
  if (NumVecs == 1)
    ReplaceUses(SDValue(N, 0), SuperReg);
  else
    for (unsigned i = 0; i < NumVecs; ++i)
      ReplaceUses(SDValue(N, i),
          CurDAG->getTargetExtractSubreg(SubRegIdx + i, dl, VT, SuperReg));

  // Update the chain
  ReplaceUses(SDValue(N, NumVecs + 1), SDValue(Ld, 2));
  return nullptr;
}

SDNode *AArch64DAGToDAGISel::SelectStore(SDNode *N, unsigned NumVecs,
                                         unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getOperand(2)->getValueType(0);

  // Form a REG_SEQUENCE to force register allocation.
  bool Is128Bit = VT.getSizeInBits() == 128;
  SmallVector<SDValue, 4> Regs(N->op_begin() + 2, N->op_begin() + 2 + NumVecs);
  SDValue RegSeq = Is128Bit ? createQTuple(Regs) : createDTuple(Regs);

  SDValue Ops[] = {RegSeq, N->getOperand(NumVecs + 2), N->getOperand(0)};
  SDNode *St = CurDAG->getMachineNode(Opc, dl, N->getValueType(0), Ops);

  return St;
}

SDNode *AArch64DAGToDAGISel::SelectPostStore(SDNode *N, unsigned NumVecs,
                                             unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getOperand(2)->getValueType(0);
  const EVT ResTys[] = {MVT::i64,    // Type of the write back register
                        MVT::Other}; // Type for the Chain

  // Form a REG_SEQUENCE to force register allocation.
  bool Is128Bit = VT.getSizeInBits() == 128;
  SmallVector<SDValue, 4> Regs(N->op_begin() + 1, N->op_begin() + 1 + NumVecs);
  SDValue RegSeq = Is128Bit ? createQTuple(Regs) : createDTuple(Regs);

  SDValue Ops[] = {RegSeq,
                   N->getOperand(NumVecs + 1), // base register
                   N->getOperand(NumVecs + 2), // Incremental
                   N->getOperand(0)};          // Chain
  SDNode *St = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);

  return St;
}

namespace {
/// WidenVector - Given a value in the V64 register class, produce the
/// equivalent value in the V128 register class.
class WidenVector {
  SelectionDAG &DAG;

public:
  WidenVector(SelectionDAG &DAG) : DAG(DAG) {}

  SDValue operator()(SDValue V64Reg) {
    EVT VT = V64Reg.getValueType();
    unsigned NarrowSize = VT.getVectorNumElements();
    MVT EltTy = VT.getVectorElementType().getSimpleVT();
    MVT WideTy = MVT::getVectorVT(EltTy, 2 * NarrowSize);
    SDLoc DL(V64Reg);

    SDValue Undef =
        SDValue(DAG.getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, WideTy), 0);
    return DAG.getTargetInsertSubreg(AArch64::dsub, DL, WideTy, Undef, V64Reg);
  }
};
} // namespace

/// NarrowVector - Given a value in the V128 register class, produce the
/// equivalent value in the V64 register class.
static SDValue NarrowVector(SDValue V128Reg, SelectionDAG &DAG) {
  EVT VT = V128Reg.getValueType();
  unsigned WideSize = VT.getVectorNumElements();
  MVT EltTy = VT.getVectorElementType().getSimpleVT();
  MVT NarrowTy = MVT::getVectorVT(EltTy, WideSize / 2);

  return DAG.getTargetExtractSubreg(AArch64::dsub, SDLoc(V128Reg), NarrowTy,
                                    V128Reg);
}

SDNode *AArch64DAGToDAGISel::SelectLoadLane(SDNode *N, unsigned NumVecs,
                                            unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  bool Narrow = VT.getSizeInBits() == 64;

  // Form a REG_SEQUENCE to force register allocation.
  SmallVector<SDValue, 4> Regs(N->op_begin() + 2, N->op_begin() + 2 + NumVecs);

  if (Narrow)
    std::transform(Regs.begin(), Regs.end(), Regs.begin(),
                   WidenVector(*CurDAG));

  SDValue RegSeq = createQTuple(Regs);

  const EVT ResTys[] = {MVT::Untyped, MVT::Other};

  unsigned LaneNo =
      cast<ConstantSDNode>(N->getOperand(NumVecs + 2))->getZExtValue();

  SDValue Ops[] = {RegSeq, CurDAG->getTargetConstant(LaneNo, dl, MVT::i64),
                   N->getOperand(NumVecs + 3), N->getOperand(0)};
  SDNode *Ld = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);
  SDValue SuperReg = SDValue(Ld, 0);

  EVT WideVT = RegSeq.getOperand(1)->getValueType(0);
  static const unsigned QSubs[] = { AArch64::qsub0, AArch64::qsub1,
                                    AArch64::qsub2, AArch64::qsub3 };
  for (unsigned i = 0; i < NumVecs; ++i) {
    SDValue NV = CurDAG->getTargetExtractSubreg(QSubs[i], dl, WideVT, SuperReg);
    if (Narrow)
      NV = NarrowVector(NV, *CurDAG);
    ReplaceUses(SDValue(N, i), NV);
  }

  ReplaceUses(SDValue(N, NumVecs), SDValue(Ld, 1));

  return Ld;
}

SDNode *AArch64DAGToDAGISel::SelectPostLoadLane(SDNode *N, unsigned NumVecs,
                                                unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getValueType(0);
  bool Narrow = VT.getSizeInBits() == 64;

  // Form a REG_SEQUENCE to force register allocation.
  SmallVector<SDValue, 4> Regs(N->op_begin() + 1, N->op_begin() + 1 + NumVecs);

  if (Narrow)
    std::transform(Regs.begin(), Regs.end(), Regs.begin(),
                   WidenVector(*CurDAG));

  SDValue RegSeq = createQTuple(Regs);

  const EVT ResTys[] = {MVT::i64, // Type of the write back register
                        RegSeq->getValueType(0), MVT::Other};

  unsigned LaneNo =
      cast<ConstantSDNode>(N->getOperand(NumVecs + 1))->getZExtValue();

  SDValue Ops[] = {RegSeq,
                   CurDAG->getTargetConstant(LaneNo, dl,
                                             MVT::i64),         // Lane Number
                   N->getOperand(NumVecs + 2),                  // Base register
                   N->getOperand(NumVecs + 3),                  // Incremental
                   N->getOperand(0)};
  SDNode *Ld = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);

  // Update uses of the write back register
  ReplaceUses(SDValue(N, NumVecs), SDValue(Ld, 0));

  // Update uses of the vector list
  SDValue SuperReg = SDValue(Ld, 1);
  if (NumVecs == 1) {
    ReplaceUses(SDValue(N, 0),
                Narrow ? NarrowVector(SuperReg, *CurDAG) : SuperReg);
  } else {
    EVT WideVT = RegSeq.getOperand(1)->getValueType(0);
    static const unsigned QSubs[] = { AArch64::qsub0, AArch64::qsub1,
                                      AArch64::qsub2, AArch64::qsub3 };
    for (unsigned i = 0; i < NumVecs; ++i) {
      SDValue NV = CurDAG->getTargetExtractSubreg(QSubs[i], dl, WideVT,
                                                  SuperReg);
      if (Narrow)
        NV = NarrowVector(NV, *CurDAG);
      ReplaceUses(SDValue(N, i), NV);
    }
  }

  // Update the Chain
  ReplaceUses(SDValue(N, NumVecs + 1), SDValue(Ld, 2));

  return Ld;
}

SDNode *AArch64DAGToDAGISel::SelectStoreLane(SDNode *N, unsigned NumVecs,
                                             unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getOperand(2)->getValueType(0);
  bool Narrow = VT.getSizeInBits() == 64;

  // Form a REG_SEQUENCE to force register allocation.
  SmallVector<SDValue, 4> Regs(N->op_begin() + 2, N->op_begin() + 2 + NumVecs);

  if (Narrow)
    std::transform(Regs.begin(), Regs.end(), Regs.begin(),
                   WidenVector(*CurDAG));

  SDValue RegSeq = createQTuple(Regs);

  unsigned LaneNo =
      cast<ConstantSDNode>(N->getOperand(NumVecs + 2))->getZExtValue();

  SDValue Ops[] = {RegSeq, CurDAG->getTargetConstant(LaneNo, dl, MVT::i64),
                   N->getOperand(NumVecs + 3), N->getOperand(0)};
  SDNode *St = CurDAG->getMachineNode(Opc, dl, MVT::Other, Ops);

  // Transfer memoperands.
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();
  cast<MachineSDNode>(St)->setMemRefs(MemOp, MemOp + 1);

  return St;
}

SDNode *AArch64DAGToDAGISel::SelectPostStoreLane(SDNode *N, unsigned NumVecs,
                                                 unsigned Opc) {
  SDLoc dl(N);
  EVT VT = N->getOperand(2)->getValueType(0);
  bool Narrow = VT.getSizeInBits() == 64;

  // Form a REG_SEQUENCE to force register allocation.
  SmallVector<SDValue, 4> Regs(N->op_begin() + 1, N->op_begin() + 1 + NumVecs);

  if (Narrow)
    std::transform(Regs.begin(), Regs.end(), Regs.begin(),
                   WidenVector(*CurDAG));

  SDValue RegSeq = createQTuple(Regs);

  const EVT ResTys[] = {MVT::i64, // Type of the write back register
                        MVT::Other};

  unsigned LaneNo =
      cast<ConstantSDNode>(N->getOperand(NumVecs + 1))->getZExtValue();

  SDValue Ops[] = {RegSeq, CurDAG->getTargetConstant(LaneNo, dl, MVT::i64),
                   N->getOperand(NumVecs + 2), // Base Register
                   N->getOperand(NumVecs + 3), // Incremental
                   N->getOperand(0)};
  SDNode *St = CurDAG->getMachineNode(Opc, dl, ResTys, Ops);

  // Transfer memoperands.
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();
  cast<MachineSDNode>(St)->setMemRefs(MemOp, MemOp + 1);

  return St;
}

static bool isBitfieldExtractOpFromAnd(SelectionDAG *CurDAG, SDNode *N,
                                       unsigned &Opc, SDValue &Opd0,
                                       unsigned &LSB, unsigned &MSB,
                                       unsigned NumberOfIgnoredLowBits,
                                       bool BiggerPattern) {
  assert(N->getOpcode() == ISD::AND &&
         "N must be a AND operation to call this function");

  EVT VT = N->getValueType(0);

  // Here we can test the type of VT and return false when the type does not
  // match, but since it is done prior to that call in the current context
  // we turned that into an assert to avoid redundant code.
  assert((VT == MVT::i32 || VT == MVT::i64) &&
         "Type checking must have been done before calling this function");

  // FIXME: simplify-demanded-bits in DAGCombine will probably have
  // changed the AND node to a 32-bit mask operation. We'll have to
  // undo that as part of the transform here if we want to catch all
  // the opportunities.
  // Currently the NumberOfIgnoredLowBits argument helps to recover
  // form these situations when matching bigger pattern (bitfield insert).

  // For unsigned extracts, check for a shift right and mask
  uint64_t And_imm = 0;
  if (!isOpcWithIntImmediate(N, ISD::AND, And_imm))
    return false;

  const SDNode *Op0 = N->getOperand(0).getNode();

  // Because of simplify-demanded-bits in DAGCombine, the mask may have been
  // simplified. Try to undo that
  And_imm |= (1 << NumberOfIgnoredLowBits) - 1;

  // The immediate is a mask of the low bits iff imm & (imm+1) == 0
  if (And_imm & (And_imm + 1))
    return false;

  bool ClampMSB = false;
  uint64_t Srl_imm = 0;
  // Handle the SRL + ANY_EXTEND case.
  if (VT == MVT::i64 && Op0->getOpcode() == ISD::ANY_EXTEND &&
      isOpcWithIntImmediate(Op0->getOperand(0).getNode(), ISD::SRL, Srl_imm)) {
    // Extend the incoming operand of the SRL to 64-bit.
    Opd0 = Widen(CurDAG, Op0->getOperand(0).getOperand(0));
    // Make sure to clamp the MSB so that we preserve the semantics of the
    // original operations.
    ClampMSB = true;
  } else if (VT == MVT::i32 && Op0->getOpcode() == ISD::TRUNCATE &&
             isOpcWithIntImmediate(Op0->getOperand(0).getNode(), ISD::SRL,
                                   Srl_imm)) {
    // If the shift result was truncated, we can still combine them.
    Opd0 = Op0->getOperand(0).getOperand(0);

    // Use the type of SRL node.
    VT = Opd0->getValueType(0);
  } else if (isOpcWithIntImmediate(Op0, ISD::SRL, Srl_imm)) {
    Opd0 = Op0->getOperand(0);
  } else if (BiggerPattern) {
    // Let's pretend a 0 shift right has been performed.
    // The resulting code will be at least as good as the original one
    // plus it may expose more opportunities for bitfield insert pattern.
    // FIXME: Currently we limit this to the bigger pattern, because
    // some optimizations expect AND and not UBFM.
    Opd0 = N->getOperand(0);
  } else
    return false;

  // Bail out on large immediates. This happens when no proper
  // combining/constant folding was performed.
  if (!BiggerPattern && (Srl_imm <= 0 || Srl_imm >= VT.getSizeInBits())) {
    DEBUG((dbgs() << N
           << ": Found large shift immediate, this should not happen\n"));
    return false;
  }

  LSB = Srl_imm;
  MSB = Srl_imm + (VT == MVT::i32 ? countTrailingOnes<uint32_t>(And_imm)
                                  : countTrailingOnes<uint64_t>(And_imm)) -
        1;
  if (ClampMSB)
    // Since we're moving the extend before the right shift operation, we need
    // to clamp the MSB to make sure we don't shift in undefined bits instead of
    // the zeros which would get shifted in with the original right shift
    // operation.
    MSB = MSB > 31 ? 31 : MSB;

  Opc = VT == MVT::i32 ? AArch64::UBFMWri : AArch64::UBFMXri;
  return true;
}

static bool isSeveralBitsExtractOpFromShr(SDNode *N, unsigned &Opc,
                                          SDValue &Opd0, unsigned &LSB,
                                          unsigned &MSB) {
  // We are looking for the following pattern which basically extracts several
  // continuous bits from the source value and places it from the LSB of the
  // destination value, all other bits of the destination value or set to zero:
  //
  // Value2 = AND Value, MaskImm
  // SRL Value2, ShiftImm
  //
  // with MaskImm >> ShiftImm to search for the bit width.
  //
  // This gets selected into a single UBFM:
  //
  // UBFM Value, ShiftImm, BitWide + Srl_imm -1
  //

  if (N->getOpcode() != ISD::SRL)
    return false;

  uint64_t And_mask = 0;
  if (!isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::AND, And_mask))
    return false;

  Opd0 = N->getOperand(0).getOperand(0);

  uint64_t Srl_imm = 0;
  if (!isIntImmediate(N->getOperand(1), Srl_imm))
    return false;

  // Check whether we really have several bits extract here.
  unsigned BitWide = 64 - countLeadingOnes(~(And_mask >> Srl_imm));
  if (BitWide && isMask_64(And_mask >> Srl_imm)) {
    if (N->getValueType(0) == MVT::i32)
      Opc = AArch64::UBFMWri;
    else
      Opc = AArch64::UBFMXri;

    LSB = Srl_imm;
    MSB = BitWide + Srl_imm - 1;
    return true;
  }

  return false;
}

static bool isBitfieldExtractOpFromShr(SDNode *N, unsigned &Opc, SDValue &Opd0,
                                       unsigned &Immr, unsigned &Imms,
                                       bool BiggerPattern) {
  assert((N->getOpcode() == ISD::SRA || N->getOpcode() == ISD::SRL) &&
         "N must be a SHR/SRA operation to call this function");

  EVT VT = N->getValueType(0);

  // Here we can test the type of VT and return false when the type does not
  // match, but since it is done prior to that call in the current context
  // we turned that into an assert to avoid redundant code.
  assert((VT == MVT::i32 || VT == MVT::i64) &&
         "Type checking must have been done before calling this function");

  // Check for AND + SRL doing several bits extract.
  if (isSeveralBitsExtractOpFromShr(N, Opc, Opd0, Immr, Imms))
    return true;

  // we're looking for a shift of a shift
  uint64_t Shl_imm = 0;
  uint64_t Trunc_bits = 0;
  if (isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::SHL, Shl_imm)) {
    Opd0 = N->getOperand(0).getOperand(0);
  } else if (VT == MVT::i32 && N->getOpcode() == ISD::SRL &&
             N->getOperand(0).getNode()->getOpcode() == ISD::TRUNCATE) {
    // We are looking for a shift of truncate. Truncate from i64 to i32 could
    // be considered as setting high 32 bits as zero. Our strategy here is to
    // always generate 64bit UBFM. This consistency will help the CSE pass
    // later find more redundancy.
    Opd0 = N->getOperand(0).getOperand(0);
    Trunc_bits = Opd0->getValueType(0).getSizeInBits() - VT.getSizeInBits();
    VT = Opd0->getValueType(0);
    assert(VT == MVT::i64 && "the promoted type should be i64");
  } else if (BiggerPattern) {
    // Let's pretend a 0 shift left has been performed.
    // FIXME: Currently we limit this to the bigger pattern case,
    // because some optimizations expect AND and not UBFM
    Opd0 = N->getOperand(0);
  } else
    return false;

  // Missing combines/constant folding may have left us with strange
  // constants.
  if (Shl_imm >= VT.getSizeInBits()) {
    DEBUG((dbgs() << N
           << ": Found large shift immediate, this should not happen\n"));
    return false;
  }

  uint64_t Srl_imm = 0;
  if (!isIntImmediate(N->getOperand(1), Srl_imm))
    return false;

  assert(Srl_imm > 0 && Srl_imm < VT.getSizeInBits() &&
         "bad amount in shift node!");
  int immr = Srl_imm - Shl_imm;
  Immr = immr < 0 ? immr + VT.getSizeInBits() : immr;
  Imms = VT.getSizeInBits() - Shl_imm - Trunc_bits - 1;
  // SRA requires a signed extraction
  if (VT == MVT::i32)
    Opc = N->getOpcode() == ISD::SRA ? AArch64::SBFMWri : AArch64::UBFMWri;
  else
    Opc = N->getOpcode() == ISD::SRA ? AArch64::SBFMXri : AArch64::UBFMXri;
  return true;
}

static bool isBitfieldExtractOp(SelectionDAG *CurDAG, SDNode *N, unsigned &Opc,
                                SDValue &Opd0, unsigned &Immr, unsigned &Imms,
                                unsigned NumberOfIgnoredLowBits = 0,
                                bool BiggerPattern = false) {
  if (N->getValueType(0) != MVT::i32 && N->getValueType(0) != MVT::i64)
    return false;

  switch (N->getOpcode()) {
  default:
    if (!N->isMachineOpcode())
      return false;
    break;
  case ISD::AND:
    return isBitfieldExtractOpFromAnd(CurDAG, N, Opc, Opd0, Immr, Imms,
                                      NumberOfIgnoredLowBits, BiggerPattern);
  case ISD::SRL:
  case ISD::SRA:
    return isBitfieldExtractOpFromShr(N, Opc, Opd0, Immr, Imms, BiggerPattern);
  }

  unsigned NOpc = N->getMachineOpcode();
  switch (NOpc) {
  default:
    return false;
  case AArch64::SBFMWri:
  case AArch64::UBFMWri:
  case AArch64::SBFMXri:
  case AArch64::UBFMXri:
    Opc = NOpc;
    Opd0 = N->getOperand(0);
    Immr = cast<ConstantSDNode>(N->getOperand(1).getNode())->getZExtValue();
    Imms = cast<ConstantSDNode>(N->getOperand(2).getNode())->getZExtValue();
    return true;
  }
  // Unreachable
  return false;
}

SDNode *AArch64DAGToDAGISel::SelectBitfieldExtractOp(SDNode *N) {
  unsigned Opc, Immr, Imms;
  SDValue Opd0;
  if (!isBitfieldExtractOp(CurDAG, N, Opc, Opd0, Immr, Imms))
    return nullptr;

  EVT VT = N->getValueType(0);
  SDLoc dl(N);

  // If the bit extract operation is 64bit but the original type is 32bit, we
  // need to add one EXTRACT_SUBREG.
  if ((Opc == AArch64::SBFMXri || Opc == AArch64::UBFMXri) && VT == MVT::i32) {
    SDValue Ops64[] = {Opd0, CurDAG->getTargetConstant(Immr, dl, MVT::i64),
                       CurDAG->getTargetConstant(Imms, dl, MVT::i64)};

    SDNode *BFM = CurDAG->getMachineNode(Opc, dl, MVT::i64, Ops64);
    SDValue SubReg = CurDAG->getTargetConstant(AArch64::sub_32, dl, MVT::i32);
    MachineSDNode *Node =
        CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG, dl, MVT::i32,
                               SDValue(BFM, 0), SubReg);
    return Node;
  }

  SDValue Ops[] = {Opd0, CurDAG->getTargetConstant(Immr, dl, VT),
                   CurDAG->getTargetConstant(Imms, dl, VT)};
  return CurDAG->SelectNodeTo(N, Opc, VT, Ops);
}

/// Does DstMask form a complementary pair with the mask provided by
/// BitsToBeInserted, suitable for use in a BFI instruction. Roughly speaking,
/// this asks whether DstMask zeroes precisely those bits that will be set by
/// the other half.
static bool isBitfieldDstMask(uint64_t DstMask, APInt BitsToBeInserted,
                              unsigned NumberOfIgnoredHighBits, EVT VT) {
  assert((VT == MVT::i32 || VT == MVT::i64) &&
         "i32 or i64 mask type expected!");
  unsigned BitWidth = VT.getSizeInBits() - NumberOfIgnoredHighBits;

  APInt SignificantDstMask = APInt(BitWidth, DstMask);
  APInt SignificantBitsToBeInserted = BitsToBeInserted.zextOrTrunc(BitWidth);

  return (SignificantDstMask & SignificantBitsToBeInserted) == 0 &&
         (SignificantDstMask | SignificantBitsToBeInserted).isAllOnesValue();
}

// Look for bits that will be useful for later uses.
// A bit is consider useless as soon as it is dropped and never used
// before it as been dropped.
// E.g., looking for useful bit of x
// 1. y = x & 0x7
// 2. z = y >> 2
// After #1, x useful bits are 0x7, then the useful bits of x, live through
// y.
// After #2, the useful bits of x are 0x4.
// However, if x is used on an unpredicatable instruction, then all its bits
// are useful.
// E.g.
// 1. y = x & 0x7
// 2. z = y >> 2
// 3. str x, [@x]
static void getUsefulBits(SDValue Op, APInt &UsefulBits, unsigned Depth = 0);

static void getUsefulBitsFromAndWithImmediate(SDValue Op, APInt &UsefulBits,
                                              unsigned Depth) {
  uint64_t Imm =
      cast<const ConstantSDNode>(Op.getOperand(1).getNode())->getZExtValue();
  Imm = AArch64_AM::decodeLogicalImmediate(Imm, UsefulBits.getBitWidth());
  UsefulBits &= APInt(UsefulBits.getBitWidth(), Imm);
  getUsefulBits(Op, UsefulBits, Depth + 1);
}

static void getUsefulBitsFromBitfieldMoveOpd(SDValue Op, APInt &UsefulBits,
                                             uint64_t Imm, uint64_t MSB,
                                             unsigned Depth) {
  // inherit the bitwidth value
  APInt OpUsefulBits(UsefulBits);
  OpUsefulBits = 1;

  if (MSB >= Imm) {
    OpUsefulBits = OpUsefulBits.shl(MSB - Imm + 1);
    --OpUsefulBits;
    // The interesting part will be in the lower part of the result
    getUsefulBits(Op, OpUsefulBits, Depth + 1);
    // The interesting part was starting at Imm in the argument
    OpUsefulBits = OpUsefulBits.shl(Imm);
  } else {
    OpUsefulBits = OpUsefulBits.shl(MSB + 1);
    --OpUsefulBits;
    // The interesting part will be shifted in the result
    OpUsefulBits = OpUsefulBits.shl(OpUsefulBits.getBitWidth() - Imm);
    getUsefulBits(Op, OpUsefulBits, Depth + 1);
    // The interesting part was at zero in the argument
    OpUsefulBits = OpUsefulBits.lshr(OpUsefulBits.getBitWidth() - Imm);
  }

  UsefulBits &= OpUsefulBits;
}

static void getUsefulBitsFromUBFM(SDValue Op, APInt &UsefulBits,
                                  unsigned Depth) {
  uint64_t Imm =
      cast<const ConstantSDNode>(Op.getOperand(1).getNode())->getZExtValue();
  uint64_t MSB =
      cast<const ConstantSDNode>(Op.getOperand(2).getNode())->getZExtValue();

  getUsefulBitsFromBitfieldMoveOpd(Op, UsefulBits, Imm, MSB, Depth);
}

static void getUsefulBitsFromOrWithShiftedReg(SDValue Op, APInt &UsefulBits,
                                              unsigned Depth) {
  uint64_t ShiftTypeAndValue =
      cast<const ConstantSDNode>(Op.getOperand(2).getNode())->getZExtValue();
  APInt Mask(UsefulBits);
  Mask.clearAllBits();
  Mask.flipAllBits();

  if (AArch64_AM::getShiftType(ShiftTypeAndValue) == AArch64_AM::LSL) {
    // Shift Left
    uint64_t ShiftAmt = AArch64_AM::getShiftValue(ShiftTypeAndValue);
    Mask = Mask.shl(ShiftAmt);
    getUsefulBits(Op, Mask, Depth + 1);
    Mask = Mask.lshr(ShiftAmt);
  } else if (AArch64_AM::getShiftType(ShiftTypeAndValue) == AArch64_AM::LSR) {
    // Shift Right
    // We do not handle AArch64_AM::ASR, because the sign will change the
    // number of useful bits
    uint64_t ShiftAmt = AArch64_AM::getShiftValue(ShiftTypeAndValue);
    Mask = Mask.lshr(ShiftAmt);
    getUsefulBits(Op, Mask, Depth + 1);
    Mask = Mask.shl(ShiftAmt);
  } else
    return;

  UsefulBits &= Mask;
}

static void getUsefulBitsFromBFM(SDValue Op, SDValue Orig, APInt &UsefulBits,
                                 unsigned Depth) {
  uint64_t Imm =
      cast<const ConstantSDNode>(Op.getOperand(2).getNode())->getZExtValue();
  uint64_t MSB =
      cast<const ConstantSDNode>(Op.getOperand(3).getNode())->getZExtValue();

  if (Op.getOperand(1) == Orig)
    return getUsefulBitsFromBitfieldMoveOpd(Op, UsefulBits, Imm, MSB, Depth);

  APInt OpUsefulBits(UsefulBits);
  OpUsefulBits = 1;

  if (MSB >= Imm) {
    OpUsefulBits = OpUsefulBits.shl(MSB - Imm + 1);
    --OpUsefulBits;
    UsefulBits &= ~OpUsefulBits;
    getUsefulBits(Op, UsefulBits, Depth + 1);
  } else {
    OpUsefulBits = OpUsefulBits.shl(MSB + 1);
    --OpUsefulBits;
    UsefulBits = ~(OpUsefulBits.shl(OpUsefulBits.getBitWidth() - Imm));
    getUsefulBits(Op, UsefulBits, Depth + 1);
  }
}

static void getUsefulBitsForUse(SDNode *UserNode, APInt &UsefulBits,
                                SDValue Orig, unsigned Depth) {

  // Users of this node should have already been instruction selected
  // FIXME: Can we turn that into an assert?
  if (!UserNode->isMachineOpcode())
    return;

  switch (UserNode->getMachineOpcode()) {
  default:
    return;
  case AArch64::ANDSWri:
  case AArch64::ANDSXri:
  case AArch64::ANDWri:
  case AArch64::ANDXri:
    // We increment Depth only when we call the getUsefulBits
    return getUsefulBitsFromAndWithImmediate(SDValue(UserNode, 0), UsefulBits,
                                             Depth);
  case AArch64::UBFMWri:
  case AArch64::UBFMXri:
    return getUsefulBitsFromUBFM(SDValue(UserNode, 0), UsefulBits, Depth);

  case AArch64::ORRWrs:
  case AArch64::ORRXrs:
    if (UserNode->getOperand(1) != Orig)
      return;
    return getUsefulBitsFromOrWithShiftedReg(SDValue(UserNode, 0), UsefulBits,
                                             Depth);
  case AArch64::BFMWri:
  case AArch64::BFMXri:
    return getUsefulBitsFromBFM(SDValue(UserNode, 0), Orig, UsefulBits, Depth);
  }
}

static void getUsefulBits(SDValue Op, APInt &UsefulBits, unsigned Depth) {
  if (Depth >= 6)
    return;
  // Initialize UsefulBits
  if (!Depth) {
    unsigned Bitwidth = Op.getValueType().getScalarType().getSizeInBits();
    // At the beginning, assume every produced bits is useful
    UsefulBits = APInt(Bitwidth, 0);
    UsefulBits.flipAllBits();
  }
  APInt UsersUsefulBits(UsefulBits.getBitWidth(), 0);

  for (SDNode *Node : Op.getNode()->uses()) {
    // A use cannot produce useful bits
    APInt UsefulBitsForUse = APInt(UsefulBits);
    getUsefulBitsForUse(Node, UsefulBitsForUse, Op, Depth);
    UsersUsefulBits |= UsefulBitsForUse;
  }
  // UsefulBits contains the produced bits that are meaningful for the
  // current definition, thus a user cannot make a bit meaningful at
  // this point
  UsefulBits &= UsersUsefulBits;
}

/// Create a machine node performing a notional SHL of Op by ShlAmount. If
/// ShlAmount is negative, do a (logical) right-shift instead. If ShlAmount is
/// 0, return Op unchanged.
static SDValue getLeftShift(SelectionDAG *CurDAG, SDValue Op, int ShlAmount) {
  if (ShlAmount == 0)
    return Op;

  EVT VT = Op.getValueType();
  SDLoc dl(Op);
  unsigned BitWidth = VT.getSizeInBits();
  unsigned UBFMOpc = BitWidth == 32 ? AArch64::UBFMWri : AArch64::UBFMXri;

  SDNode *ShiftNode;
  if (ShlAmount > 0) {
    // LSL wD, wN, #Amt == UBFM wD, wN, #32-Amt, #31-Amt
    ShiftNode = CurDAG->getMachineNode(
        UBFMOpc, dl, VT, Op,
        CurDAG->getTargetConstant(BitWidth - ShlAmount, dl, VT),
        CurDAG->getTargetConstant(BitWidth - 1 - ShlAmount, dl, VT));
  } else {
    // LSR wD, wN, #Amt == UBFM wD, wN, #Amt, #32-1
    assert(ShlAmount < 0 && "expected right shift");
    int ShrAmount = -ShlAmount;
    ShiftNode = CurDAG->getMachineNode(
        UBFMOpc, dl, VT, Op, CurDAG->getTargetConstant(ShrAmount, dl, VT),
        CurDAG->getTargetConstant(BitWidth - 1, dl, VT));
  }

  return SDValue(ShiftNode, 0);
}

/// Does this tree qualify as an attempt to move a bitfield into position,
/// essentially "(and (shl VAL, N), Mask)".
static bool isBitfieldPositioningOp(SelectionDAG *CurDAG, SDValue Op,
                                    bool BiggerPattern,
                                    SDValue &Src, int &ShiftAmount,
                                    int &MaskWidth) {
  EVT VT = Op.getValueType();
  unsigned BitWidth = VT.getSizeInBits();
  (void)BitWidth;
  assert(BitWidth == 32 || BitWidth == 64);

  APInt KnownZero, KnownOne;
  CurDAG->computeKnownBits(Op, KnownZero, KnownOne);

  // Non-zero in the sense that they're not provably zero, which is the key
  // point if we want to use this value
  uint64_t NonZeroBits = (~KnownZero).getZExtValue();

  // Discard a constant AND mask if present. It's safe because the node will
  // already have been factored into the computeKnownBits calculation above.
  uint64_t AndImm;
  if (isOpcWithIntImmediate(Op.getNode(), ISD::AND, AndImm)) {
    assert((~APInt(BitWidth, AndImm) & ~KnownZero) == 0);
    Op = Op.getOperand(0);
  }

  // Don't match if the SHL has more than one use, since then we'll end up
  // generating SHL+UBFIZ instead of just keeping SHL+AND.
  if (!BiggerPattern && !Op.hasOneUse())
    return false;

  uint64_t ShlImm;
  if (!isOpcWithIntImmediate(Op.getNode(), ISD::SHL, ShlImm))
    return false;
  Op = Op.getOperand(0);

  if (!isShiftedMask_64(NonZeroBits))
    return false;

  ShiftAmount = countTrailingZeros(NonZeroBits);
  MaskWidth = countTrailingOnes(NonZeroBits >> ShiftAmount);

  // BFI encompasses sufficiently many nodes that it's worth inserting an extra
  // LSL/LSR if the mask in NonZeroBits doesn't quite match up with the ISD::SHL
  // amount.  BiggerPattern is true when this pattern is being matched for BFI,
  // BiggerPattern is false when this pattern is being matched for UBFIZ, in
  // which case it is not profitable to insert an extra shift.
  if (ShlImm - ShiftAmount != 0 && !BiggerPattern)
    return false;
  Src = getLeftShift(CurDAG, Op, ShlImm - ShiftAmount);

  return true;
}

// Given a OR operation, check if we have the following pattern
// ubfm c, b, imm, imm2 (or something that does the same jobs, see
//                       isBitfieldExtractOp)
// d = e & mask2 ; where mask is a binary sequence of 1..10..0 and
//                 countTrailingZeros(mask2) == imm2 - imm + 1
// f = d | c
// if yes, given reference arguments will be update so that one can replace
// the OR instruction with:
// f = Opc Opd0, Opd1, LSB, MSB ; where Opc is a BFM, LSB = imm, and MSB = imm2
static bool isBitfieldInsertOpFromOr(SDNode *N, unsigned &Opc, SDValue &Dst,
                                     SDValue &Src, unsigned &ImmR,
                                     unsigned &ImmS, SelectionDAG *CurDAG) {
  assert(N->getOpcode() == ISD::OR && "Expect a OR operation");

  // Set Opc
  EVT VT = N->getValueType(0);
  if (VT == MVT::i32)
    Opc = AArch64::BFMWri;
  else if (VT == MVT::i64)
    Opc = AArch64::BFMXri;
  else
    return false;

  // Because of simplify-demanded-bits in DAGCombine, involved masks may not
  // have the expected shape. Try to undo that.
  APInt UsefulBits;
  getUsefulBits(SDValue(N, 0), UsefulBits);

  unsigned NumberOfIgnoredLowBits = UsefulBits.countTrailingZeros();
  unsigned NumberOfIgnoredHighBits = UsefulBits.countLeadingZeros();

  // OR is commutative, check all combinations of operand order and values of
  // BiggerPattern, i.e.
  //     Opd0, Opd1, BiggerPattern=false
  //     Opd1, Opd0, BiggerPattern=false
  //     Opd0, Opd1, BiggerPattern=true
  //     Opd1, Opd0, BiggerPattern=true
  // Several of these combinations may match, so check with BiggerPattern=false
  // first since that will produce better results by matching more instructions
  // and/or inserting fewer extra instructions.
  for (int I = 0; I < 4; ++I) {

    bool BiggerPattern = I / 2;
    SDNode *OrOpd0 = N->getOperand(I % 2).getNode();
    SDValue OrOpd1Val = N->getOperand((I + 1) % 2);
    SDNode *OrOpd1 = OrOpd1Val.getNode();

    unsigned BFXOpc;
    int DstLSB, Width;
    if (isBitfieldExtractOp(CurDAG, OrOpd0, BFXOpc, Src, ImmR, ImmS,
                            NumberOfIgnoredLowBits, BiggerPattern)) {
      // Check that the returned opcode is compatible with the pattern,
      // i.e., same type and zero extended (U and not S)
      if ((BFXOpc != AArch64::UBFMXri && VT == MVT::i64) ||
          (BFXOpc != AArch64::UBFMWri && VT == MVT::i32))
        continue;

      // Compute the width of the bitfield insertion
      DstLSB = 0;
      Width = ImmS - ImmR + 1;
      // FIXME: This constraint is to catch bitfield insertion we may
      // want to widen the pattern if we want to grab general bitfied
      // move case
      if (Width <= 0)
        continue;

      // If the mask on the insertee is correct, we have a BFXIL operation. We
      // can share the ImmR and ImmS values from the already-computed UBFM.
    } else if (isBitfieldPositioningOp(CurDAG, SDValue(OrOpd0, 0),
                                       BiggerPattern,
                                       Src, DstLSB, Width)) {
      ImmR = (VT.getSizeInBits() - DstLSB) % VT.getSizeInBits();
      ImmS = Width - 1;
    } else
      continue;

    // Check the second part of the pattern
    EVT VT = OrOpd1->getValueType(0);
    assert((VT == MVT::i32 || VT == MVT::i64) && "unexpected OR operand");

    // Compute the Known Zero for the candidate of the first operand.
    // This allows to catch more general case than just looking for
    // AND with imm. Indeed, simplify-demanded-bits may have removed
    // the AND instruction because it proves it was useless.
    APInt KnownZero, KnownOne;
    CurDAG->computeKnownBits(OrOpd1Val, KnownZero, KnownOne);

    // Check if there is enough room for the second operand to appear
    // in the first one
    APInt BitsToBeInserted =
        APInt::getBitsSet(KnownZero.getBitWidth(), DstLSB, DstLSB + Width);

    if ((BitsToBeInserted & ~KnownZero) != 0)
      continue;

    // Set the first operand
    uint64_t Imm;
    if (isOpcWithIntImmediate(OrOpd1, ISD::AND, Imm) &&
        isBitfieldDstMask(Imm, BitsToBeInserted, NumberOfIgnoredHighBits, VT))
      // In that case, we can eliminate the AND
      Dst = OrOpd1->getOperand(0);
    else
      // Maybe the AND has been removed by simplify-demanded-bits
      // or is useful because it discards more bits
      Dst = OrOpd1Val;

    // both parts match
    return true;
  }

  return false;
}

SDNode *AArch64DAGToDAGISel::SelectBitfieldInsertOp(SDNode *N) {
  if (N->getOpcode() != ISD::OR)
    return nullptr;

  unsigned Opc;
  unsigned LSB, MSB;
  SDValue Opd0, Opd1;

  if (!isBitfieldInsertOpFromOr(N, Opc, Opd0, Opd1, LSB, MSB, CurDAG))
    return nullptr;

  EVT VT = N->getValueType(0);
  SDLoc dl(N);
  SDValue Ops[] = { Opd0,
                    Opd1,
                    CurDAG->getTargetConstant(LSB, dl, VT),
                    CurDAG->getTargetConstant(MSB, dl, VT) };
  return CurDAG->SelectNodeTo(N, Opc, VT, Ops);
}

/// SelectBitfieldInsertInZeroOp - Match a UBFIZ instruction that is the
/// equivalent of a left shift by a constant amount followed by an and masking
/// out a contiguous set of bits.
SDNode *AArch64DAGToDAGISel::SelectBitfieldInsertInZeroOp(SDNode *N) {
  if (N->getOpcode() != ISD::AND)
    return nullptr;

  EVT VT = N->getValueType(0);
  unsigned Opc;
  if (VT == MVT::i32)
    Opc = AArch64::UBFMWri;
  else if (VT == MVT::i64)
    Opc = AArch64::UBFMXri;
  else
    return nullptr;

  SDValue Op0;
  int DstLSB, Width;
  if (!isBitfieldPositioningOp(CurDAG, SDValue(N, 0), /*BiggerPattern=*/false,
                               Op0, DstLSB, Width))
    return nullptr;

  // ImmR is the rotate right amount.
  unsigned ImmR = (VT.getSizeInBits() - DstLSB) % VT.getSizeInBits();
  // ImmS is the most significant bit of the source to be moved.
  unsigned ImmS = Width - 1;

  SDLoc DL(N);
  SDValue Ops[] = {Op0, CurDAG->getTargetConstant(ImmR, DL, VT),
                   CurDAG->getTargetConstant(ImmS, DL, VT)};
  return CurDAG->SelectNodeTo(N, Opc, VT, Ops);
}

bool
AArch64DAGToDAGISel::SelectCVTFixedPosOperand(SDValue N, SDValue &FixedPos,
                                              unsigned RegWidth) {
  APFloat FVal(0.0);
  if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(N))
    FVal = CN->getValueAPF();
  else if (LoadSDNode *LN = dyn_cast<LoadSDNode>(N)) {
    // Some otherwise illegal constants are allowed in this case.
    if (LN->getOperand(1).getOpcode() != AArch64ISD::ADDlow ||
        !isa<ConstantPoolSDNode>(LN->getOperand(1)->getOperand(1)))
      return false;

    ConstantPoolSDNode *CN =
        dyn_cast<ConstantPoolSDNode>(LN->getOperand(1)->getOperand(1));
    FVal = cast<ConstantFP>(CN->getConstVal())->getValueAPF();
  } else
    return false;

  // An FCVT[SU] instruction performs: convertToInt(Val * 2^fbits) where fbits
  // is between 1 and 32 for a destination w-register, or 1 and 64 for an
  // x-register.
  //
  // By this stage, we've detected (fp_to_[su]int (fmul Val, THIS_NODE)) so we
  // want THIS_NODE to be 2^fbits. This is much easier to deal with using
  // integers.
  bool IsExact;

  // fbits is between 1 and 64 in the worst-case, which means the fmul
  // could have 2^64 as an actual operand. Need 65 bits of precision.
  APSInt IntVal(65, true);
  FVal.convertToInteger(IntVal, APFloat::rmTowardZero, &IsExact);

  // N.b. isPowerOf2 also checks for > 0.
  if (!IsExact || !IntVal.isPowerOf2()) return false;
  unsigned FBits = IntVal.logBase2();

  // Checks above should have guaranteed that we haven't lost information in
  // finding FBits, but it must still be in range.
  if (FBits == 0 || FBits > RegWidth) return false;

  FixedPos = CurDAG->getTargetConstant(FBits, SDLoc(N), MVT::i32);
  return true;
}

// Inspects a register string of the form o0:op1:CRn:CRm:op2 gets the fields
// of the string and obtains the integer values from them and combines these
// into a single value to be used in the MRS/MSR instruction.
static int getIntOperandFromRegisterString(StringRef RegString) {
  SmallVector<StringRef, 5> Fields;
  RegString.split(Fields, ':');

  if (Fields.size() == 1)
    return -1;

  assert(Fields.size() == 5
            && "Invalid number of fields in read register string");

  SmallVector<int, 5> Ops;
  bool AllIntFields = true;

  for (StringRef Field : Fields) {
    unsigned IntField;
    AllIntFields &= !Field.getAsInteger(10, IntField);
    Ops.push_back(IntField);
  }

  assert(AllIntFields &&
          "Unexpected non-integer value in special register string.");

  // Need to combine the integer fields of the string into a single value
  // based on the bit encoding of MRS/MSR instruction.
  return (Ops[0] << 14) | (Ops[1] << 11) | (Ops[2] << 7) |
         (Ops[3] << 3) | (Ops[4]);
}

// Lower the read_register intrinsic to an MRS instruction node if the special
// register string argument is either of the form detailed in the ALCE (the
// form described in getIntOperandsFromRegsterString) or is a named register
// known by the MRS SysReg mapper.
SDNode *AArch64DAGToDAGISel::SelectReadRegister(SDNode *N) {
  const MDNodeSDNode *MD = dyn_cast<MDNodeSDNode>(N->getOperand(1));
  const MDString *RegString = dyn_cast<MDString>(MD->getMD()->getOperand(0));
  SDLoc DL(N);

  int Reg = getIntOperandFromRegisterString(RegString->getString());
  if (Reg != -1)
    return CurDAG->getMachineNode(AArch64::MRS, DL, N->getSimpleValueType(0),
                                  MVT::Other,
                                  CurDAG->getTargetConstant(Reg, DL, MVT::i32),
                                  N->getOperand(0));

  // Use the sysreg mapper to map the remaining possible strings to the
  // value for the register to be used for the instruction operand.
  AArch64SysReg::MRSMapper mapper;
  bool IsValidSpecialReg;
  Reg = mapper.fromString(RegString->getString(),
                          Subtarget->getFeatureBits(),
                          IsValidSpecialReg);
  if (IsValidSpecialReg)
    return CurDAG->getMachineNode(AArch64::MRS, DL, N->getSimpleValueType(0),
                                  MVT::Other,
                                  CurDAG->getTargetConstant(Reg, DL, MVT::i32),
                                  N->getOperand(0));

  return nullptr;
}

// Lower the write_register intrinsic to an MSR instruction node if the special
// register string argument is either of the form detailed in the ALCE (the
// form described in getIntOperandsFromRegsterString) or is a named register
// known by the MSR SysReg mapper.
SDNode *AArch64DAGToDAGISel::SelectWriteRegister(SDNode *N) {
  const MDNodeSDNode *MD = dyn_cast<MDNodeSDNode>(N->getOperand(1));
  const MDString *RegString = dyn_cast<MDString>(MD->getMD()->getOperand(0));
  SDLoc DL(N);

  int Reg = getIntOperandFromRegisterString(RegString->getString());
  if (Reg != -1)
    return CurDAG->getMachineNode(AArch64::MSR, DL, MVT::Other,
                                  CurDAG->getTargetConstant(Reg, DL, MVT::i32),
                                  N->getOperand(2), N->getOperand(0));

  // Check if the register was one of those allowed as the pstatefield value in
  // the MSR (immediate) instruction. To accept the values allowed in the
  // pstatefield for the MSR (immediate) instruction, we also require that an
  // immediate value has been provided as an argument, we know that this is
  // the case as it has been ensured by semantic checking.
  AArch64PState::PStateMapper PMapper;
  bool IsValidSpecialReg;
  Reg = PMapper.fromString(RegString->getString(),
                           Subtarget->getFeatureBits(),
                           IsValidSpecialReg);
  if (IsValidSpecialReg) {
    assert (isa<ConstantSDNode>(N->getOperand(2))
              && "Expected a constant integer expression.");
    uint64_t Immed = cast<ConstantSDNode>(N->getOperand(2))->getZExtValue();
    unsigned State;
    if (Reg == AArch64PState::PAN) {
      assert(Immed < 2 && "Bad imm");
      State = AArch64::MSRpstateImm1;
    } else {
      assert(Immed < 16 && "Bad imm");
      State = AArch64::MSRpstateImm4;
    }
    return CurDAG->getMachineNode(State, DL, MVT::Other,
                                  CurDAG->getTargetConstant(Reg, DL, MVT::i32),
                                  CurDAG->getTargetConstant(Immed, DL, MVT::i16),
                                  N->getOperand(0));
  }

  // Use the sysreg mapper to attempt to map the remaining possible strings
  // to the value for the register to be used for the MSR (register)
  // instruction operand.
  AArch64SysReg::MSRMapper Mapper;
  Reg = Mapper.fromString(RegString->getString(),
                          Subtarget->getFeatureBits(),
                          IsValidSpecialReg);

  if (IsValidSpecialReg)
    return CurDAG->getMachineNode(AArch64::MSR, DL, MVT::Other,
                                  CurDAG->getTargetConstant(Reg, DL, MVT::i32),
                                  N->getOperand(2), N->getOperand(0));

  return nullptr;
}

SDNode *AArch64DAGToDAGISel::Select(SDNode *Node) {
  // Dump information about the Node being selected
  DEBUG(errs() << "Selecting: ");
  DEBUG(Node->dump(CurDAG));
  DEBUG(errs() << "\n");

  // If we have a custom node, we already have selected!
  if (Node->isMachineOpcode()) {
    DEBUG(errs() << "== "; Node->dump(CurDAG); errs() << "\n");
    Node->setNodeId(-1);
    return nullptr;
  }

  // Few custom selection stuff.
  SDNode *ResNode = nullptr;
  EVT VT = Node->getValueType(0);

  switch (Node->getOpcode()) {
  default:
    break;

  case ISD::READ_REGISTER:
    if (SDNode *Res = SelectReadRegister(Node))
      return Res;
    break;

  case ISD::WRITE_REGISTER:
    if (SDNode *Res = SelectWriteRegister(Node))
      return Res;
    break;

  case ISD::ADD:
    if (SDNode *I = SelectMLAV64LaneV128(Node))
      return I;
    break;

  case ISD::LOAD: {
    // Try to select as an indexed load. Fall through to normal processing
    // if we can't.
    bool Done = false;
    SDNode *I = SelectIndexedLoad(Node, Done);
    if (Done)
      return I;
    break;
  }

  case ISD::SRL:
  case ISD::AND:
  case ISD::SRA:
    if (SDNode *I = SelectBitfieldExtractOp(Node))
      return I;
    if (SDNode *I = SelectBitfieldInsertInZeroOp(Node))
      return I;
    break;

  case ISD::OR:
    if (SDNode *I = SelectBitfieldInsertOp(Node))
      return I;
    break;

  case ISD::EXTRACT_VECTOR_ELT: {
    // Extracting lane zero is a special case where we can just use a plain
    // EXTRACT_SUBREG instruction, which will become FMOV. This is easier for
    // the rest of the compiler, especially the register allocator and copyi
    // propagation, to reason about, so is preferred when it's possible to
    // use it.
    ConstantSDNode *LaneNode = cast<ConstantSDNode>(Node->getOperand(1));
    // Bail and use the default Select() for non-zero lanes.
    if (LaneNode->getZExtValue() != 0)
      break;
    // If the element type is not the same as the result type, likewise
    // bail and use the default Select(), as there's more to do than just
    // a cross-class COPY. This catches extracts of i8 and i16 elements
    // since they will need an explicit zext.
    if (VT != Node->getOperand(0).getValueType().getVectorElementType())
      break;
    unsigned SubReg;
    switch (Node->getOperand(0)
                .getValueType()
                .getVectorElementType()
                .getSizeInBits()) {
    default:
      llvm_unreachable("Unexpected vector element type!");
    case 64:
      SubReg = AArch64::dsub;
      break;
    case 32:
      SubReg = AArch64::ssub;
      break;
    case 16:
      SubReg = AArch64::hsub;
      break;
    case 8:
      llvm_unreachable("unexpected zext-requiring extract element!");
    }
    SDValue Extract = CurDAG->getTargetExtractSubreg(SubReg, SDLoc(Node), VT,
                                                     Node->getOperand(0));
    DEBUG(dbgs() << "ISEL: Custom selection!\n=> ");
    DEBUG(Extract->dumpr(CurDAG));
    DEBUG(dbgs() << "\n");
    return Extract.getNode();
  }
  case ISD::Constant: {
    // Materialize zero constants as copies from WZR/XZR.  This allows
    // the coalescer to propagate these into other instructions.
    ConstantSDNode *ConstNode = cast<ConstantSDNode>(Node);
    if (ConstNode->isNullValue()) {
      if (VT == MVT::i32)
        return CurDAG->getCopyFromReg(CurDAG->getEntryNode(), SDLoc(Node),
                                      AArch64::WZR, MVT::i32).getNode();
      else if (VT == MVT::i64)
        return CurDAG->getCopyFromReg(CurDAG->getEntryNode(), SDLoc(Node),
                                      AArch64::XZR, MVT::i64).getNode();
    }
    break;
  }

  case ISD::FrameIndex: {
    // Selects to ADDXri FI, 0 which in turn will become ADDXri SP, imm.
    int FI = cast<FrameIndexSDNode>(Node)->getIndex();
    unsigned Shifter = AArch64_AM::getShifterImm(AArch64_AM::LSL, 0);
    const TargetLowering *TLI = getTargetLowering();
    SDValue TFI = CurDAG->getTargetFrameIndex(
        FI, TLI->getPointerTy(CurDAG->getDataLayout()));
    SDLoc DL(Node);
    SDValue Ops[] = { TFI, CurDAG->getTargetConstant(0, DL, MVT::i32),
                      CurDAG->getTargetConstant(Shifter, DL, MVT::i32) };
    return CurDAG->SelectNodeTo(Node, AArch64::ADDXri, MVT::i64, Ops);
  }
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    switch (IntNo) {
    default:
      break;
    case Intrinsic::aarch64_ldaxp:
    case Intrinsic::aarch64_ldxp: {
      unsigned Op =
          IntNo == Intrinsic::aarch64_ldaxp ? AArch64::LDAXPX : AArch64::LDXPX;
      SDValue MemAddr = Node->getOperand(2);
      SDLoc DL(Node);
      SDValue Chain = Node->getOperand(0);

      SDNode *Ld = CurDAG->getMachineNode(Op, DL, MVT::i64, MVT::i64,
                                          MVT::Other, MemAddr, Chain);

      // Transfer memoperands.
      MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
      MemOp[0] = cast<MemIntrinsicSDNode>(Node)->getMemOperand();
      cast<MachineSDNode>(Ld)->setMemRefs(MemOp, MemOp + 1);
      return Ld;
    }
    case Intrinsic::aarch64_stlxp:
    case Intrinsic::aarch64_stxp: {
      unsigned Op =
          IntNo == Intrinsic::aarch64_stlxp ? AArch64::STLXPX : AArch64::STXPX;
      SDLoc DL(Node);
      SDValue Chain = Node->getOperand(0);
      SDValue ValLo = Node->getOperand(2);
      SDValue ValHi = Node->getOperand(3);
      SDValue MemAddr = Node->getOperand(4);

      // Place arguments in the right order.
      SDValue Ops[] = {ValLo, ValHi, MemAddr, Chain};

      SDNode *St = CurDAG->getMachineNode(Op, DL, MVT::i32, MVT::Other, Ops);
      // Transfer memoperands.
      MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
      MemOp[0] = cast<MemIntrinsicSDNode>(Node)->getMemOperand();
      cast<MachineSDNode>(St)->setMemRefs(MemOp, MemOp + 1);

      return St;
    }
    case Intrinsic::aarch64_neon_ld1x2:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 2, AArch64::LD1Twov8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 2, AArch64::LD1Twov16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 2, AArch64::LD1Twov4h, AArch64::dsub0);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectLoad(Node, 2, AArch64::LD1Twov8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 2, AArch64::LD1Twov2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 2, AArch64::LD1Twov4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 2, AArch64::LD1Twov1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 2, AArch64::LD1Twov2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld1x3:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 3, AArch64::LD1Threev8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 3, AArch64::LD1Threev16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 3, AArch64::LD1Threev4h, AArch64::dsub0);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectLoad(Node, 3, AArch64::LD1Threev8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 3, AArch64::LD1Threev2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 3, AArch64::LD1Threev4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 3, AArch64::LD1Threev1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 3, AArch64::LD1Threev2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld1x4:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 4, AArch64::LD1Fourv8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 4, AArch64::LD1Fourv16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 4, AArch64::LD1Fourv4h, AArch64::dsub0);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectLoad(Node, 4, AArch64::LD1Fourv8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 4, AArch64::LD1Fourv2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 4, AArch64::LD1Fourv4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 4, AArch64::LD1Fourv1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 4, AArch64::LD1Fourv2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld2:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 2, AArch64::LD2Twov8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 2, AArch64::LD2Twov16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 2, AArch64::LD2Twov4h, AArch64::dsub0);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectLoad(Node, 2, AArch64::LD2Twov8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 2, AArch64::LD2Twov2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 2, AArch64::LD2Twov4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 2, AArch64::LD1Twov1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 2, AArch64::LD2Twov2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld3:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 3, AArch64::LD3Threev8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 3, AArch64::LD3Threev16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 3, AArch64::LD3Threev4h, AArch64::dsub0);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectLoad(Node, 3, AArch64::LD3Threev8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 3, AArch64::LD3Threev2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 3, AArch64::LD3Threev4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 3, AArch64::LD1Threev1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 3, AArch64::LD3Threev2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld4:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 4, AArch64::LD4Fourv8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 4, AArch64::LD4Fourv16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 4, AArch64::LD4Fourv4h, AArch64::dsub0);
      else if (VT == MVT::v8i16  || VT == MVT::v8f16)
        return SelectLoad(Node, 4, AArch64::LD4Fourv8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 4, AArch64::LD4Fourv2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 4, AArch64::LD4Fourv4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 4, AArch64::LD1Fourv1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 4, AArch64::LD4Fourv2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld2r:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 2, AArch64::LD2Rv8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 2, AArch64::LD2Rv16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 2, AArch64::LD2Rv4h, AArch64::dsub0);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectLoad(Node, 2, AArch64::LD2Rv8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 2, AArch64::LD2Rv2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 2, AArch64::LD2Rv4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 2, AArch64::LD2Rv1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 2, AArch64::LD2Rv2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld3r:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 3, AArch64::LD3Rv8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 3, AArch64::LD3Rv16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 3, AArch64::LD3Rv4h, AArch64::dsub0);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectLoad(Node, 3, AArch64::LD3Rv8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 3, AArch64::LD3Rv2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 3, AArch64::LD3Rv4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 3, AArch64::LD3Rv1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 3, AArch64::LD3Rv2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld4r:
      if (VT == MVT::v8i8)
        return SelectLoad(Node, 4, AArch64::LD4Rv8b, AArch64::dsub0);
      else if (VT == MVT::v16i8)
        return SelectLoad(Node, 4, AArch64::LD4Rv16b, AArch64::qsub0);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectLoad(Node, 4, AArch64::LD4Rv4h, AArch64::dsub0);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectLoad(Node, 4, AArch64::LD4Rv8h, AArch64::qsub0);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectLoad(Node, 4, AArch64::LD4Rv2s, AArch64::dsub0);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectLoad(Node, 4, AArch64::LD4Rv4s, AArch64::qsub0);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectLoad(Node, 4, AArch64::LD4Rv1d, AArch64::dsub0);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectLoad(Node, 4, AArch64::LD4Rv2d, AArch64::qsub0);
      break;
    case Intrinsic::aarch64_neon_ld2lane:
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectLoadLane(Node, 2, AArch64::LD2i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
               VT == MVT::v8f16)
        return SelectLoadLane(Node, 2, AArch64::LD2i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectLoadLane(Node, 2, AArch64::LD2i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectLoadLane(Node, 2, AArch64::LD2i64);
      break;
    case Intrinsic::aarch64_neon_ld3lane:
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectLoadLane(Node, 3, AArch64::LD3i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
               VT == MVT::v8f16)
        return SelectLoadLane(Node, 3, AArch64::LD3i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectLoadLane(Node, 3, AArch64::LD3i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectLoadLane(Node, 3, AArch64::LD3i64);
      break;
    case Intrinsic::aarch64_neon_ld4lane:
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectLoadLane(Node, 4, AArch64::LD4i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
               VT == MVT::v8f16)
        return SelectLoadLane(Node, 4, AArch64::LD4i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectLoadLane(Node, 4, AArch64::LD4i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectLoadLane(Node, 4, AArch64::LD4i64);
      break;
    }
  } break;
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(0))->getZExtValue();
    switch (IntNo) {
    default:
      break;
    case Intrinsic::aarch64_neon_tbl2:
      return SelectTable(Node, 2, VT == MVT::v8i8 ? AArch64::TBLv8i8Two
                                                  : AArch64::TBLv16i8Two,
                         false);
    case Intrinsic::aarch64_neon_tbl3:
      return SelectTable(Node, 3, VT == MVT::v8i8 ? AArch64::TBLv8i8Three
                                                  : AArch64::TBLv16i8Three,
                         false);
    case Intrinsic::aarch64_neon_tbl4:
      return SelectTable(Node, 4, VT == MVT::v8i8 ? AArch64::TBLv8i8Four
                                                  : AArch64::TBLv16i8Four,
                         false);
    case Intrinsic::aarch64_neon_tbx2:
      return SelectTable(Node, 2, VT == MVT::v8i8 ? AArch64::TBXv8i8Two
                                                  : AArch64::TBXv16i8Two,
                         true);
    case Intrinsic::aarch64_neon_tbx3:
      return SelectTable(Node, 3, VT == MVT::v8i8 ? AArch64::TBXv8i8Three
                                                  : AArch64::TBXv16i8Three,
                         true);
    case Intrinsic::aarch64_neon_tbx4:
      return SelectTable(Node, 4, VT == MVT::v8i8 ? AArch64::TBXv8i8Four
                                                  : AArch64::TBXv16i8Four,
                         true);
    case Intrinsic::aarch64_neon_smull:
    case Intrinsic::aarch64_neon_umull:
      if (SDNode *N = SelectMULLV64LaneV128(IntNo, Node))
        return N;
      break;
    }
    break;
  }
  case ISD::INTRINSIC_VOID: {
    unsigned IntNo = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    if (Node->getNumOperands() >= 3)
      VT = Node->getOperand(2)->getValueType(0);
    switch (IntNo) {
    default:
      break;
    case Intrinsic::aarch64_neon_st1x2: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 2, AArch64::ST1Twov8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 2, AArch64::ST1Twov16b);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectStore(Node, 2, AArch64::ST1Twov4h);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectStore(Node, 2, AArch64::ST1Twov8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 2, AArch64::ST1Twov2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 2, AArch64::ST1Twov4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 2, AArch64::ST1Twov2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 2, AArch64::ST1Twov1d);
      break;
    }
    case Intrinsic::aarch64_neon_st1x3: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 3, AArch64::ST1Threev8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 3, AArch64::ST1Threev16b);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectStore(Node, 3, AArch64::ST1Threev4h);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectStore(Node, 3, AArch64::ST1Threev8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 3, AArch64::ST1Threev2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 3, AArch64::ST1Threev4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 3, AArch64::ST1Threev2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 3, AArch64::ST1Threev1d);
      break;
    }
    case Intrinsic::aarch64_neon_st1x4: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 4, AArch64::ST1Fourv8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 4, AArch64::ST1Fourv16b);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectStore(Node, 4, AArch64::ST1Fourv4h);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectStore(Node, 4, AArch64::ST1Fourv8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 4, AArch64::ST1Fourv2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 4, AArch64::ST1Fourv4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 4, AArch64::ST1Fourv2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 4, AArch64::ST1Fourv1d);
      break;
    }
    case Intrinsic::aarch64_neon_st2: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 2, AArch64::ST2Twov8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 2, AArch64::ST2Twov16b);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectStore(Node, 2, AArch64::ST2Twov4h);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectStore(Node, 2, AArch64::ST2Twov8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 2, AArch64::ST2Twov2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 2, AArch64::ST2Twov4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 2, AArch64::ST2Twov2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 2, AArch64::ST1Twov1d);
      break;
    }
    case Intrinsic::aarch64_neon_st3: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 3, AArch64::ST3Threev8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 3, AArch64::ST3Threev16b);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectStore(Node, 3, AArch64::ST3Threev4h);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectStore(Node, 3, AArch64::ST3Threev8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 3, AArch64::ST3Threev2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 3, AArch64::ST3Threev4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 3, AArch64::ST3Threev2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 3, AArch64::ST1Threev1d);
      break;
    }
    case Intrinsic::aarch64_neon_st4: {
      if (VT == MVT::v8i8)
        return SelectStore(Node, 4, AArch64::ST4Fourv8b);
      else if (VT == MVT::v16i8)
        return SelectStore(Node, 4, AArch64::ST4Fourv16b);
      else if (VT == MVT::v4i16 || VT == MVT::v4f16)
        return SelectStore(Node, 4, AArch64::ST4Fourv4h);
      else if (VT == MVT::v8i16 || VT == MVT::v8f16)
        return SelectStore(Node, 4, AArch64::ST4Fourv8h);
      else if (VT == MVT::v2i32 || VT == MVT::v2f32)
        return SelectStore(Node, 4, AArch64::ST4Fourv2s);
      else if (VT == MVT::v4i32 || VT == MVT::v4f32)
        return SelectStore(Node, 4, AArch64::ST4Fourv4s);
      else if (VT == MVT::v2i64 || VT == MVT::v2f64)
        return SelectStore(Node, 4, AArch64::ST4Fourv2d);
      else if (VT == MVT::v1i64 || VT == MVT::v1f64)
        return SelectStore(Node, 4, AArch64::ST1Fourv1d);
      break;
    }
    case Intrinsic::aarch64_neon_st2lane: {
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectStoreLane(Node, 2, AArch64::ST2i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
               VT == MVT::v8f16)
        return SelectStoreLane(Node, 2, AArch64::ST2i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectStoreLane(Node, 2, AArch64::ST2i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectStoreLane(Node, 2, AArch64::ST2i64);
      break;
    }
    case Intrinsic::aarch64_neon_st3lane: {
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectStoreLane(Node, 3, AArch64::ST3i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
               VT == MVT::v8f16)
        return SelectStoreLane(Node, 3, AArch64::ST3i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectStoreLane(Node, 3, AArch64::ST3i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectStoreLane(Node, 3, AArch64::ST3i64);
      break;
    }
    case Intrinsic::aarch64_neon_st4lane: {
      if (VT == MVT::v16i8 || VT == MVT::v8i8)
        return SelectStoreLane(Node, 4, AArch64::ST4i8);
      else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
               VT == MVT::v8f16)
        return SelectStoreLane(Node, 4, AArch64::ST4i16);
      else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
               VT == MVT::v2f32)
        return SelectStoreLane(Node, 4, AArch64::ST4i32);
      else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
               VT == MVT::v1f64)
        return SelectStoreLane(Node, 4, AArch64::ST4i64);
      break;
    }
    }
    break;
  }
  case AArch64ISD::LD2post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 2, AArch64::LD2Twov8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 2, AArch64::LD2Twov16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 2, AArch64::LD2Twov4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 2, AArch64::LD2Twov8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 2, AArch64::LD2Twov2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 2, AArch64::LD2Twov4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 2, AArch64::LD2Twov2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD3post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 3, AArch64::LD3Threev8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 3, AArch64::LD3Threev16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 3, AArch64::LD3Threev4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 3, AArch64::LD3Threev8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 3, AArch64::LD3Threev2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 3, AArch64::LD3Threev4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 3, AArch64::LD3Threev2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD4post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 4, AArch64::LD4Fourv8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 4, AArch64::LD4Fourv16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 4, AArch64::LD4Fourv4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 4, AArch64::LD4Fourv8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 4, AArch64::LD4Fourv2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 4, AArch64::LD4Fourv4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 4, AArch64::LD4Fourv2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD1x2post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 2, AArch64::LD1Twov2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD1x3post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 3, AArch64::LD1Threev2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD1x4post: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 4, AArch64::LD1Fourv2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD1DUPpost: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 1, AArch64::LD1Rv8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 1, AArch64::LD1Rv16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 1, AArch64::LD1Rv4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 1, AArch64::LD1Rv8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 1, AArch64::LD1Rv2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 1, AArch64::LD1Rv4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 1, AArch64::LD1Rv1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 1, AArch64::LD1Rv2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD2DUPpost: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 2, AArch64::LD2Rv8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 2, AArch64::LD2Rv16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 2, AArch64::LD2Rv4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 2, AArch64::LD2Rv8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 2, AArch64::LD2Rv2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 2, AArch64::LD2Rv4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 2, AArch64::LD2Rv1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 2, AArch64::LD2Rv2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD3DUPpost: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 3, AArch64::LD3Rv8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 3, AArch64::LD3Rv16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 3, AArch64::LD3Rv4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 3, AArch64::LD3Rv8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 3, AArch64::LD3Rv2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 3, AArch64::LD3Rv4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 3, AArch64::LD3Rv1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 3, AArch64::LD3Rv2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD4DUPpost: {
    if (VT == MVT::v8i8)
      return SelectPostLoad(Node, 4, AArch64::LD4Rv8b_POST, AArch64::dsub0);
    else if (VT == MVT::v16i8)
      return SelectPostLoad(Node, 4, AArch64::LD4Rv16b_POST, AArch64::qsub0);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostLoad(Node, 4, AArch64::LD4Rv4h_POST, AArch64::dsub0);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostLoad(Node, 4, AArch64::LD4Rv8h_POST, AArch64::qsub0);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostLoad(Node, 4, AArch64::LD4Rv2s_POST, AArch64::dsub0);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostLoad(Node, 4, AArch64::LD4Rv4s_POST, AArch64::qsub0);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostLoad(Node, 4, AArch64::LD4Rv1d_POST, AArch64::dsub0);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostLoad(Node, 4, AArch64::LD4Rv2d_POST, AArch64::qsub0);
    break;
  }
  case AArch64ISD::LD1LANEpost: {
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostLoadLane(Node, 1, AArch64::LD1i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
             VT == MVT::v8f16)
      return SelectPostLoadLane(Node, 1, AArch64::LD1i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostLoadLane(Node, 1, AArch64::LD1i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostLoadLane(Node, 1, AArch64::LD1i64_POST);
    break;
  }
  case AArch64ISD::LD2LANEpost: {
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostLoadLane(Node, 2, AArch64::LD2i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
             VT == MVT::v8f16)
      return SelectPostLoadLane(Node, 2, AArch64::LD2i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostLoadLane(Node, 2, AArch64::LD2i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostLoadLane(Node, 2, AArch64::LD2i64_POST);
    break;
  }
  case AArch64ISD::LD3LANEpost: {
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostLoadLane(Node, 3, AArch64::LD3i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
             VT == MVT::v8f16)
      return SelectPostLoadLane(Node, 3, AArch64::LD3i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostLoadLane(Node, 3, AArch64::LD3i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostLoadLane(Node, 3, AArch64::LD3i64_POST);
    break;
  }
  case AArch64ISD::LD4LANEpost: {
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostLoadLane(Node, 4, AArch64::LD4i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
             VT == MVT::v8f16)
      return SelectPostLoadLane(Node, 4, AArch64::LD4i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostLoadLane(Node, 4, AArch64::LD4i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostLoadLane(Node, 4, AArch64::LD4i64_POST);
    break;
  }
  case AArch64ISD::ST2post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 2, AArch64::ST2Twov8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 2, AArch64::ST2Twov16b_POST);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostStore(Node, 2, AArch64::ST2Twov4h_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostStore(Node, 2, AArch64::ST2Twov8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 2, AArch64::ST2Twov2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 2, AArch64::ST2Twov4s_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 2, AArch64::ST2Twov2d_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 2, AArch64::ST1Twov1d_POST);
    break;
  }
  case AArch64ISD::ST3post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 3, AArch64::ST3Threev8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 3, AArch64::ST3Threev16b_POST);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostStore(Node, 3, AArch64::ST3Threev4h_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostStore(Node, 3, AArch64::ST3Threev8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 3, AArch64::ST3Threev2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 3, AArch64::ST3Threev4s_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 3, AArch64::ST3Threev2d_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 3, AArch64::ST1Threev1d_POST);
    break;
  }
  case AArch64ISD::ST4post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 4, AArch64::ST4Fourv8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 4, AArch64::ST4Fourv16b_POST);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostStore(Node, 4, AArch64::ST4Fourv4h_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostStore(Node, 4, AArch64::ST4Fourv8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 4, AArch64::ST4Fourv2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 4, AArch64::ST4Fourv4s_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 4, AArch64::ST4Fourv2d_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv1d_POST);
    break;
  }
  case AArch64ISD::ST1x2post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 2, AArch64::ST1Twov8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 2, AArch64::ST1Twov16b_POST);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostStore(Node, 2, AArch64::ST1Twov4h_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostStore(Node, 2, AArch64::ST1Twov8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 2, AArch64::ST1Twov2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 2, AArch64::ST1Twov4s_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 2, AArch64::ST1Twov1d_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 2, AArch64::ST1Twov2d_POST);
    break;
  }
  case AArch64ISD::ST1x3post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 3, AArch64::ST1Threev8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 3, AArch64::ST1Threev16b_POST);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostStore(Node, 3, AArch64::ST1Threev4h_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostStore(Node, 3, AArch64::ST1Threev8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 3, AArch64::ST1Threev2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 3, AArch64::ST1Threev4s_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 3, AArch64::ST1Threev1d_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 3, AArch64::ST1Threev2d_POST);
    break;
  }
  case AArch64ISD::ST1x4post: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v8i8)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv8b_POST);
    else if (VT == MVT::v16i8)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv16b_POST);
    else if (VT == MVT::v4i16 || VT == MVT::v4f16)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv4h_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v8f16)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv8h_POST);
    else if (VT == MVT::v2i32 || VT == MVT::v2f32)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv2s_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v4f32)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv4s_POST);
    else if (VT == MVT::v1i64 || VT == MVT::v1f64)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv1d_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v2f64)
      return SelectPostStore(Node, 4, AArch64::ST1Fourv2d_POST);
    break;
  }
  case AArch64ISD::ST2LANEpost: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostStoreLane(Node, 2, AArch64::ST2i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
             VT == MVT::v8f16)
      return SelectPostStoreLane(Node, 2, AArch64::ST2i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostStoreLane(Node, 2, AArch64::ST2i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostStoreLane(Node, 2, AArch64::ST2i64_POST);
    break;
  }
  case AArch64ISD::ST3LANEpost: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostStoreLane(Node, 3, AArch64::ST3i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
             VT == MVT::v8f16)
      return SelectPostStoreLane(Node, 3, AArch64::ST3i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostStoreLane(Node, 3, AArch64::ST3i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostStoreLane(Node, 3, AArch64::ST3i64_POST);
    break;
  }
  case AArch64ISD::ST4LANEpost: {
    VT = Node->getOperand(1).getValueType();
    if (VT == MVT::v16i8 || VT == MVT::v8i8)
      return SelectPostStoreLane(Node, 4, AArch64::ST4i8_POST);
    else if (VT == MVT::v8i16 || VT == MVT::v4i16 || VT == MVT::v4f16 ||
             VT == MVT::v8f16)
      return SelectPostStoreLane(Node, 4, AArch64::ST4i16_POST);
    else if (VT == MVT::v4i32 || VT == MVT::v2i32 || VT == MVT::v4f32 ||
             VT == MVT::v2f32)
      return SelectPostStoreLane(Node, 4, AArch64::ST4i32_POST);
    else if (VT == MVT::v2i64 || VT == MVT::v1i64 || VT == MVT::v2f64 ||
             VT == MVT::v1f64)
      return SelectPostStoreLane(Node, 4, AArch64::ST4i64_POST);
    break;
  }
  }

  // Select the default instruction
  ResNode = SelectCode(Node);

  DEBUG(errs() << "=> ");
  if (ResNode == nullptr || ResNode == Node)
    DEBUG(Node->dump(CurDAG));
  else
    DEBUG(ResNode->dump(CurDAG));
  DEBUG(errs() << "\n");

  return ResNode;
}

/// createAArch64ISelDag - This pass converts a legalized DAG into a
/// AArch64-specific DAG, ready for instruction scheduling.
FunctionPass *llvm::createAArch64ISelDag(AArch64TargetMachine &TM,
                                         CodeGenOpt::Level OptLevel) {
  return new AArch64DAGToDAGISel(TM, OptLevel);
}
