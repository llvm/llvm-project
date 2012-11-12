//===-- ARMISelDAGToDAG.cpp - A dag to dag inst selector for ARM ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the ARM target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-isel"
#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMTargetMachine.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<bool>
DisableShifterOp("disable-shifter-op", cl::Hidden,
  cl::desc("Disable isel of shifter-op"),
  cl::init(false));

static cl::opt<bool>
CheckVMLxHazard("check-vmlx-hazard", cl::Hidden,
  cl::desc("Check fp vmla / vmls hazard at isel time"),
  cl::init(true));

//===--------------------------------------------------------------------===//
/// ARMDAGToDAGISel - ARM specific code to select ARM machine
/// instructions for SelectionDAG operations.
///
namespace {

enum AddrMode2Type {
  AM2_BASE, // Simple AM2 (+-imm12)
  AM2_SHOP  // Shifter-op AM2
};

class ARMDAGToDAGISel : public SelectionDAGISel {
  ARMBaseTargetMachine &TM;
  const ARMBaseInstrInfo *TII;

  /// Subtarget - Keep a pointer to the ARMSubtarget around so that we can
  /// make the right decision when generating code for different targets.
  const ARMSubtarget *Subtarget;

public:
  explicit ARMDAGToDAGISel(ARMBaseTargetMachine &tm,
                           CodeGenOpt::Level OptLevel)
    : SelectionDAGISel(tm, OptLevel), TM(tm),
      TII(static_cast<const ARMBaseInstrInfo*>(TM.getInstrInfo())),
      Subtarget(&TM.getSubtarget<ARMSubtarget>()) {
  }

  virtual const char *getPassName() const {
    return "ARM Instruction Selection";
  }

  /// getI32Imm - Return a target constant of type i32 with the specified
  /// value.
  inline SDValue getI32Imm(unsigned Imm) {
    return CurDAG->getTargetConstant(Imm, MVT::i32);
  }

  SDNode *Select(SDNode *N);


  bool hasNoVMLxHazardUse(SDNode *N) const;
  bool isShifterOpProfitable(const SDValue &Shift,
                             ARM_AM::ShiftOpc ShOpcVal, unsigned ShAmt);
  bool SelectRegShifterOperand(SDValue N, SDValue &A,
                               SDValue &B, SDValue &C,
                               bool CheckProfitability = true);
  bool SelectImmShifterOperand(SDValue N, SDValue &A,
                               SDValue &B, bool CheckProfitability = true);
  bool SelectShiftRegShifterOperand(SDValue N, SDValue &A,
                                    SDValue &B, SDValue &C) {
    // Don't apply the profitability check
    return SelectRegShifterOperand(N, A, B, C, false);
  }
  bool SelectShiftImmShifterOperand(SDValue N, SDValue &A,
                                    SDValue &B) {
    // Don't apply the profitability check
    return SelectImmShifterOperand(N, A, B, false);
  }

  bool SelectAddrModeImm12(SDValue N, SDValue &Base, SDValue &OffImm);
  bool SelectLdStSOReg(SDValue N, SDValue &Base, SDValue &Offset, SDValue &Opc);

  AddrMode2Type SelectAddrMode2Worker(SDValue N, SDValue &Base,
                                      SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode2Base(SDValue N, SDValue &Base, SDValue &Offset,
                           SDValue &Opc) {
    return SelectAddrMode2Worker(N, Base, Offset, Opc) == AM2_BASE;
  }

  bool SelectAddrMode2ShOp(SDValue N, SDValue &Base, SDValue &Offset,
                           SDValue &Opc) {
    return SelectAddrMode2Worker(N, Base, Offset, Opc) == AM2_SHOP;
  }

  bool SelectAddrMode2(SDValue N, SDValue &Base, SDValue &Offset,
                       SDValue &Opc) {
    SelectAddrMode2Worker(N, Base, Offset, Opc);
//    return SelectAddrMode2ShOp(N, Base, Offset, Opc);
    // This always matches one way or another.
    return true;
  }

  bool SelectAddrMode2OffsetReg(SDNode *Op, SDValue N,
                             SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode2OffsetImm(SDNode *Op, SDValue N,
                             SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode2OffsetImmPre(SDNode *Op, SDValue N,
                             SDValue &Offset, SDValue &Opc);
  bool SelectAddrOffsetNone(SDValue N, SDValue &Base);
  bool SelectAddrMode3(SDValue N, SDValue &Base,
                       SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode3Offset(SDNode *Op, SDValue N,
                             SDValue &Offset, SDValue &Opc);
  bool SelectAddrMode5(SDValue N, SDValue &Base,
                       SDValue &Offset);
  bool SelectAddrMode6(SDNode *Parent, SDValue N, SDValue &Addr,SDValue &Align);
  bool SelectAddrMode6Offset(SDNode *Op, SDValue N, SDValue &Offset);

  bool SelectAddrModePC(SDValue N, SDValue &Offset, SDValue &Label);

  // Thumb Addressing Modes:
  bool SelectThumbAddrModeRR(SDValue N, SDValue &Base, SDValue &Offset);
  bool SelectThumbAddrModeRI(SDValue N, SDValue &Base, SDValue &Offset,
                             unsigned Scale);
  bool SelectThumbAddrModeRI5S1(SDValue N, SDValue &Base, SDValue &Offset);
  bool SelectThumbAddrModeRI5S2(SDValue N, SDValue &Base, SDValue &Offset);
  bool SelectThumbAddrModeRI5S4(SDValue N, SDValue &Base, SDValue &Offset);
  bool SelectThumbAddrModeImm5S(SDValue N, unsigned Scale, SDValue &Base,
                                SDValue &OffImm);
  bool SelectThumbAddrModeImm5S1(SDValue N, SDValue &Base,
                                 SDValue &OffImm);
  bool SelectThumbAddrModeImm5S2(SDValue N, SDValue &Base,
                                 SDValue &OffImm);
  bool SelectThumbAddrModeImm5S4(SDValue N, SDValue &Base,
                                 SDValue &OffImm);
  bool SelectThumbAddrModeSP(SDValue N, SDValue &Base, SDValue &OffImm);

  // Thumb 2 Addressing Modes:
  bool SelectT2ShifterOperandReg(SDValue N,
                                 SDValue &BaseReg, SDValue &Opc);
  bool SelectT2AddrModeImm12(SDValue N, SDValue &Base, SDValue &OffImm);
  bool SelectT2AddrModeImm8(SDValue N, SDValue &Base,
                            SDValue &OffImm);
  bool SelectT2AddrModeImm8Offset(SDNode *Op, SDValue N,
                                 SDValue &OffImm);
  bool SelectT2AddrModeSoReg(SDValue N, SDValue &Base,
                             SDValue &OffReg, SDValue &ShImm);

  inline bool is_so_imm(unsigned Imm) const {
    return ARM_AM::getSOImmVal(Imm) != -1;
  }

  inline bool is_so_imm_not(unsigned Imm) const {
    return ARM_AM::getSOImmVal(~Imm) != -1;
  }

  inline bool is_t2_so_imm(unsigned Imm) const {
    return ARM_AM::getT2SOImmVal(Imm) != -1;
  }

  inline bool is_t2_so_imm_not(unsigned Imm) const {
    return ARM_AM::getT2SOImmVal(~Imm) != -1;
  }

  // Include the pieces autogenerated from the target description.
#include "ARMGenDAGISel.inc"

private:
  /// SelectARMIndexedLoad - Indexed (pre/post inc/dec) load matching code for
  /// ARM.
  SDNode *SelectARMIndexedLoad(SDNode *N);
  SDNode *SelectT2IndexedLoad(SDNode *N);

  /// SelectVLD - Select NEON load intrinsics.  NumVecs should be
  /// 1, 2, 3 or 4.  The opcode arrays specify the instructions used for
  /// loads of D registers and even subregs and odd subregs of Q registers.
  /// For NumVecs <= 2, QOpcodes1 is not used.
  SDNode *SelectVLD(SDNode *N, bool isUpdating, unsigned NumVecs,
                    const uint16_t *DOpcodes,
                    const uint16_t *QOpcodes0, const uint16_t *QOpcodes1);

  /// SelectVST - Select NEON store intrinsics.  NumVecs should
  /// be 1, 2, 3 or 4.  The opcode arrays specify the instructions used for
  /// stores of D registers and even subregs and odd subregs of Q registers.
  /// For NumVecs <= 2, QOpcodes1 is not used.
  SDNode *SelectVST(SDNode *N, bool isUpdating, unsigned NumVecs,
                    const uint16_t *DOpcodes,
                    const uint16_t *QOpcodes0, const uint16_t *QOpcodes1);

  /// SelectVLDSTLane - Select NEON load/store lane intrinsics.  NumVecs should
  /// be 2, 3 or 4.  The opcode arrays specify the instructions used for
  /// load/store of D registers and Q registers.
  SDNode *SelectVLDSTLane(SDNode *N, bool IsLoad,
                          bool isUpdating, unsigned NumVecs,
                          const uint16_t *DOpcodes, const uint16_t *QOpcodes);

  /// SelectVLDDup - Select NEON load-duplicate intrinsics.  NumVecs
  /// should be 2, 3 or 4.  The opcode array specifies the instructions used
  /// for loading D registers.  (Q registers are not supported.)
  SDNode *SelectVLDDup(SDNode *N, bool isUpdating, unsigned NumVecs,
                       const uint16_t *Opcodes);

  /// SelectVTBL - Select NEON VTBL and VTBX intrinsics.  NumVecs should be 2,
  /// 3 or 4.  These are custom-selected so that a REG_SEQUENCE can be
  /// generated to force the table registers to be consecutive.
  SDNode *SelectVTBL(SDNode *N, bool IsExt, unsigned NumVecs, unsigned Opc);

  /// SelectV6T2BitfieldExtractOp - Select SBFX/UBFX instructions for ARM.
  SDNode *SelectV6T2BitfieldExtractOp(SDNode *N, bool isSigned);

  /// SelectCMOVOp - Select CMOV instructions for ARM.
  SDNode *SelectCMOVOp(SDNode *N);
  SDNode *SelectT2CMOVShiftOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                              ARMCC::CondCodes CCVal, SDValue CCR,
                              SDValue InFlag);
  SDNode *SelectARMCMOVShiftOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                               ARMCC::CondCodes CCVal, SDValue CCR,
                               SDValue InFlag);
  SDNode *SelectT2CMOVImmOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                              ARMCC::CondCodes CCVal, SDValue CCR,
                              SDValue InFlag);
  SDNode *SelectARMCMOVImmOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                               ARMCC::CondCodes CCVal, SDValue CCR,
                               SDValue InFlag);

  // Select special operations if node forms integer ABS pattern
  SDNode *SelectABSOp(SDNode *N);

  SDNode *SelectConcatVector(SDNode *N);

  SDNode *SelectAtomic64(SDNode *Node, unsigned Opc);

  /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
  /// inline asm expressions.
  virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                            char ConstraintCode,
                                            std::vector<SDValue> &OutOps);

  // Form pairs of consecutive S, D, or Q registers.
  SDNode *PairSRegs(EVT VT, SDValue V0, SDValue V1);
  SDNode *PairDRegs(EVT VT, SDValue V0, SDValue V1);
  SDNode *PairQRegs(EVT VT, SDValue V0, SDValue V1);

  // Form sequences of 4 consecutive S, D, or Q registers.
  SDNode *QuadSRegs(EVT VT, SDValue V0, SDValue V1, SDValue V2, SDValue V3);
  SDNode *QuadDRegs(EVT VT, SDValue V0, SDValue V1, SDValue V2, SDValue V3);
  SDNode *QuadQRegs(EVT VT, SDValue V0, SDValue V1, SDValue V2, SDValue V3);

  // Get the alignment operand for a NEON VLD or VST instruction.
  SDValue GetVLDSTAlign(SDValue Align, unsigned NumVecs, bool is64BitVector);
};
}

/// isInt32Immediate - This method tests to see if the node is a 32-bit constant
/// operand. If so Imm will receive the 32-bit value.
static bool isInt32Immediate(SDNode *N, unsigned &Imm) {
  if (N->getOpcode() == ISD::Constant && N->getValueType(0) == MVT::i32) {
    Imm = cast<ConstantSDNode>(N)->getZExtValue();
    return true;
  }
  return false;
}

// isInt32Immediate - This method tests to see if a constant operand.
// If so Imm will receive the 32 bit value.
static bool isInt32Immediate(SDValue N, unsigned &Imm) {
  return isInt32Immediate(N.getNode(), Imm);
}

// isOpcWithIntImmediate - This method tests to see if the node is a specific
// opcode and that it has a immediate integer right operand.
// If so Imm will receive the 32 bit value.
static bool isOpcWithIntImmediate(SDNode *N, unsigned Opc, unsigned& Imm) {
  return N->getOpcode() == Opc &&
         isInt32Immediate(N->getOperand(1).getNode(), Imm);
}

/// \brief Check whether a particular node is a constant value representable as
/// (N * Scale) where (N in [\p RangeMin, \p RangeMax).
///
/// \param ScaledConstant [out] - On success, the pre-scaled constant value.
static bool isScaledConstantInRange(SDValue Node, int Scale,
                                    int RangeMin, int RangeMax,
                                    int &ScaledConstant) {
  assert(Scale > 0 && "Invalid scale!");

  // Check that this is a constant.
  const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Node);
  if (!C)
    return false;

  ScaledConstant = (int) C->getZExtValue();
  if ((ScaledConstant % Scale) != 0)
    return false;

  ScaledConstant /= Scale;
  return ScaledConstant >= RangeMin && ScaledConstant < RangeMax;
}

/// hasNoVMLxHazardUse - Return true if it's desirable to select a FP MLA / MLS
/// node. VFP / NEON fp VMLA / VMLS instructions have special RAW hazards (at
/// least on current ARM implementations) which should be avoidded.
bool ARMDAGToDAGISel::hasNoVMLxHazardUse(SDNode *N) const {
  if (OptLevel == CodeGenOpt::None)
    return true;

  if (!CheckVMLxHazard)
    return true;

  if (!Subtarget->isCortexA8() && !Subtarget->isLikeA9() &&
      !Subtarget->isSwift())
    return true;

  if (!N->hasOneUse())
    return false;

  SDNode *Use = *N->use_begin();
  if (Use->getOpcode() == ISD::CopyToReg)
    return true;
  if (Use->isMachineOpcode()) {
    const MCInstrDesc &MCID = TII->get(Use->getMachineOpcode());
    if (MCID.mayStore())
      return true;
    unsigned Opcode = MCID.getOpcode();
    if (Opcode == ARM::VMOVRS || Opcode == ARM::VMOVRRD)
      return true;
    // vmlx feeding into another vmlx. We actually want to unfold
    // the use later in the MLxExpansion pass. e.g.
    // vmla
    // vmla (stall 8 cycles)
    //
    // vmul (5 cycles)
    // vadd (5 cycles)
    // vmla
    // This adds up to about 18 - 19 cycles.
    //
    // vmla
    // vmul (stall 4 cycles)
    // vadd adds up to about 14 cycles.
    return TII->isFpMLxInstruction(Opcode);
  }

  return false;
}

bool ARMDAGToDAGISel::isShifterOpProfitable(const SDValue &Shift,
                                            ARM_AM::ShiftOpc ShOpcVal,
                                            unsigned ShAmt) {
  if (!Subtarget->isLikeA9() && !Subtarget->isSwift())
    return true;
  if (Shift.hasOneUse())
    return true;
  // R << 2 is free.
  return ShOpcVal == ARM_AM::lsl &&
         (ShAmt == 2 || (Subtarget->isSwift() && ShAmt == 1));
}

bool ARMDAGToDAGISel::SelectImmShifterOperand(SDValue N,
                                              SDValue &BaseReg,
                                              SDValue &Opc,
                                              bool CheckProfitability) {
  if (DisableShifterOp)
    return false;

  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(N.getOpcode());

  // Don't match base register only case. That is matched to a separate
  // lower complexity pattern with explicit register operand.
  if (ShOpcVal == ARM_AM::no_shift) return false;

  BaseReg = N.getOperand(0);
  unsigned ShImmVal = 0;
  ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1));
  if (!RHS) return false;
  ShImmVal = RHS->getZExtValue() & 31;
  Opc = CurDAG->getTargetConstant(ARM_AM::getSORegOpc(ShOpcVal, ShImmVal),
                                  MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectRegShifterOperand(SDValue N,
                                              SDValue &BaseReg,
                                              SDValue &ShReg,
                                              SDValue &Opc,
                                              bool CheckProfitability) {
  if (DisableShifterOp)
    return false;

  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(N.getOpcode());

  // Don't match base register only case. That is matched to a separate
  // lower complexity pattern with explicit register operand.
  if (ShOpcVal == ARM_AM::no_shift) return false;

  BaseReg = N.getOperand(0);
  unsigned ShImmVal = 0;
  ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1));
  if (RHS) return false;

  ShReg = N.getOperand(1);
  if (CheckProfitability && !isShifterOpProfitable(N, ShOpcVal, ShImmVal))
    return false;
  Opc = CurDAG->getTargetConstant(ARM_AM::getSORegOpc(ShOpcVal, ShImmVal),
                                  MVT::i32);
  return true;
}


bool ARMDAGToDAGISel::SelectAddrModeImm12(SDValue N,
                                          SDValue &Base,
                                          SDValue &OffImm) {
  // Match simple R + imm12 operands.

  // Base only.
  if (N.getOpcode() != ISD::ADD && N.getOpcode() != ISD::SUB &&
      !CurDAG->isBaseWithConstantOffset(N)) {
    if (N.getOpcode() == ISD::FrameIndex) {
      // Match frame index.
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
      return true;
    }

    if (N.getOpcode() == ARMISD::Wrapper &&
        !(Subtarget->useMovt() &&
                     N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
    } else
      Base = N;
    OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int RHSC = (int)RHS->getZExtValue();
    if (N.getOpcode() == ISD::SUB)
      RHSC = -RHSC;

    if (RHSC >= 0 && RHSC < 0x1000) { // 12 bits (unsigned)
      Base   = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      }
      OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
      return true;
    }
  }

  // Base only.
  Base = N;
  OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}



bool ARMDAGToDAGISel::SelectLdStSOReg(SDValue N, SDValue &Base, SDValue &Offset,
                                      SDValue &Opc) {
  if (N.getOpcode() == ISD::MUL &&
      ((!Subtarget->isLikeA9() && !Subtarget->isSwift()) || N.hasOneUse())) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      // X * [3,5,9] -> X + X * [2,4,8] etc.
      int RHSC = (int)RHS->getZExtValue();
      if (RHSC & 1) {
        RHSC = RHSC & ~1;
        ARM_AM::AddrOpc AddSub = ARM_AM::add;
        if (RHSC < 0) {
          AddSub = ARM_AM::sub;
          RHSC = - RHSC;
        }
        if (isPowerOf2_32(RHSC)) {
          unsigned ShAmt = Log2_32(RHSC);
          Base = Offset = N.getOperand(0);
          Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, ShAmt,
                                                            ARM_AM::lsl),
                                          MVT::i32);
          return true;
        }
      }
    }
  }

  if (N.getOpcode() != ISD::ADD && N.getOpcode() != ISD::SUB &&
      // ISD::OR that is equivalent to an ISD::ADD.
      !CurDAG->isBaseWithConstantOffset(N))
    return false;

  // Leave simple R +/- imm12 operands for LDRi12
  if (N.getOpcode() == ISD::ADD || N.getOpcode() == ISD::OR) {
    int RHSC;
    if (isScaledConstantInRange(N.getOperand(1), /*Scale=*/1,
                                -0x1000+1, 0x1000, RHSC)) // 12 bits.
      return false;
  }

  // Otherwise this is R +/- [possibly shifted] R.
  ARM_AM::AddrOpc AddSub = N.getOpcode() == ISD::SUB ? ARM_AM::sub:ARM_AM::add;
  ARM_AM::ShiftOpc ShOpcVal =
    ARM_AM::getShiftOpcForNode(N.getOperand(1).getOpcode());
  unsigned ShAmt = 0;

  Base   = N.getOperand(0);
  Offset = N.getOperand(1);

  if (ShOpcVal != ARM_AM::no_shift) {
    // Check to see if the RHS of the shift is a constant, if not, we can't fold
    // it.
    if (ConstantSDNode *Sh =
           dyn_cast<ConstantSDNode>(N.getOperand(1).getOperand(1))) {
      ShAmt = Sh->getZExtValue();
      if (isShifterOpProfitable(Offset, ShOpcVal, ShAmt))
        Offset = N.getOperand(1).getOperand(0);
      else {
        ShAmt = 0;
        ShOpcVal = ARM_AM::no_shift;
      }
    } else {
      ShOpcVal = ARM_AM::no_shift;
    }
  }

  // Try matching (R shl C) + (R).
  if (N.getOpcode() != ISD::SUB && ShOpcVal == ARM_AM::no_shift &&
      !(Subtarget->isLikeA9() || Subtarget->isSwift() ||
        N.getOperand(0).hasOneUse())) {
    ShOpcVal = ARM_AM::getShiftOpcForNode(N.getOperand(0).getOpcode());
    if (ShOpcVal != ARM_AM::no_shift) {
      // Check to see if the RHS of the shift is a constant, if not, we can't
      // fold it.
      if (ConstantSDNode *Sh =
          dyn_cast<ConstantSDNode>(N.getOperand(0).getOperand(1))) {
        ShAmt = Sh->getZExtValue();
        if (isShifterOpProfitable(N.getOperand(0), ShOpcVal, ShAmt)) {
          Offset = N.getOperand(0).getOperand(0);
          Base = N.getOperand(1);
        } else {
          ShAmt = 0;
          ShOpcVal = ARM_AM::no_shift;
        }
      } else {
        ShOpcVal = ARM_AM::no_shift;
      }
    }
  }

  Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, ShAmt, ShOpcVal),
                                  MVT::i32);
  return true;
}


//-----

AddrMode2Type ARMDAGToDAGISel::SelectAddrMode2Worker(SDValue N,
                                                     SDValue &Base,
                                                     SDValue &Offset,
                                                     SDValue &Opc) {
  if (N.getOpcode() == ISD::MUL &&
      (!(Subtarget->isLikeA9() || Subtarget->isSwift()) || N.hasOneUse())) {
    if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      // X * [3,5,9] -> X + X * [2,4,8] etc.
      int RHSC = (int)RHS->getZExtValue();
      if (RHSC & 1) {
        RHSC = RHSC & ~1;
        ARM_AM::AddrOpc AddSub = ARM_AM::add;
        if (RHSC < 0) {
          AddSub = ARM_AM::sub;
          RHSC = - RHSC;
        }
        if (isPowerOf2_32(RHSC)) {
          unsigned ShAmt = Log2_32(RHSC);
          Base = Offset = N.getOperand(0);
          Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, ShAmt,
                                                            ARM_AM::lsl),
                                          MVT::i32);
          return AM2_SHOP;
        }
      }
    }
  }

  if (N.getOpcode() != ISD::ADD && N.getOpcode() != ISD::SUB &&
      // ISD::OR that is equivalent to an ADD.
      !CurDAG->isBaseWithConstantOffset(N)) {
    Base = N;
    if (N.getOpcode() == ISD::FrameIndex) {
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    } else if (N.getOpcode() == ARMISD::Wrapper &&
               !(Subtarget->useMovt() &&
                 N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
    }
    Offset = CurDAG->getRegister(0, MVT::i32);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(ARM_AM::add, 0,
                                                      ARM_AM::no_shift),
                                    MVT::i32);
    return AM2_BASE;
  }

  // Match simple R +/- imm12 operands.
  if (N.getOpcode() != ISD::SUB) {
    int RHSC;
    if (isScaledConstantInRange(N.getOperand(1), /*Scale=*/1,
                                -0x1000+1, 0x1000, RHSC)) { // 12 bits.
      Base = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      }
      Offset = CurDAG->getRegister(0, MVT::i32);

      ARM_AM::AddrOpc AddSub = ARM_AM::add;
      if (RHSC < 0) {
        AddSub = ARM_AM::sub;
        RHSC = - RHSC;
      }
      Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, RHSC,
                                                        ARM_AM::no_shift),
                                      MVT::i32);
      return AM2_BASE;
    }
  }

  if ((Subtarget->isLikeA9() || Subtarget->isSwift()) && !N.hasOneUse()) {
    // Compute R +/- (R << N) and reuse it.
    Base = N;
    Offset = CurDAG->getRegister(0, MVT::i32);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(ARM_AM::add, 0,
                                                      ARM_AM::no_shift),
                                    MVT::i32);
    return AM2_BASE;
  }

  // Otherwise this is R +/- [possibly shifted] R.
  ARM_AM::AddrOpc AddSub = N.getOpcode() != ISD::SUB ? ARM_AM::add:ARM_AM::sub;
  ARM_AM::ShiftOpc ShOpcVal =
    ARM_AM::getShiftOpcForNode(N.getOperand(1).getOpcode());
  unsigned ShAmt = 0;

  Base   = N.getOperand(0);
  Offset = N.getOperand(1);

  if (ShOpcVal != ARM_AM::no_shift) {
    // Check to see if the RHS of the shift is a constant, if not, we can't fold
    // it.
    if (ConstantSDNode *Sh =
           dyn_cast<ConstantSDNode>(N.getOperand(1).getOperand(1))) {
      ShAmt = Sh->getZExtValue();
      if (isShifterOpProfitable(Offset, ShOpcVal, ShAmt))
        Offset = N.getOperand(1).getOperand(0);
      else {
        ShAmt = 0;
        ShOpcVal = ARM_AM::no_shift;
      }
    } else {
      ShOpcVal = ARM_AM::no_shift;
    }
  }

  // Try matching (R shl C) + (R).
  if (N.getOpcode() != ISD::SUB && ShOpcVal == ARM_AM::no_shift &&
      !(Subtarget->isLikeA9() || Subtarget->isSwift() ||
        N.getOperand(0).hasOneUse())) {
    ShOpcVal = ARM_AM::getShiftOpcForNode(N.getOperand(0).getOpcode());
    if (ShOpcVal != ARM_AM::no_shift) {
      // Check to see if the RHS of the shift is a constant, if not, we can't
      // fold it.
      if (ConstantSDNode *Sh =
          dyn_cast<ConstantSDNode>(N.getOperand(0).getOperand(1))) {
        ShAmt = Sh->getZExtValue();
        if (isShifterOpProfitable(N.getOperand(0), ShOpcVal, ShAmt)) {
          Offset = N.getOperand(0).getOperand(0);
          Base = N.getOperand(1);
        } else {
          ShAmt = 0;
          ShOpcVal = ARM_AM::no_shift;
        }
      } else {
        ShOpcVal = ARM_AM::no_shift;
      }
    }
  }

  Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, ShAmt, ShOpcVal),
                                  MVT::i32);
  return AM2_SHOP;
}

bool ARMDAGToDAGISel::SelectAddrMode2OffsetReg(SDNode *Op, SDValue N,
                                            SDValue &Offset, SDValue &Opc) {
  unsigned Opcode = Op->getOpcode();
  ISD::MemIndexedMode AM = (Opcode == ISD::LOAD)
    ? cast<LoadSDNode>(Op)->getAddressingMode()
    : cast<StoreSDNode>(Op)->getAddressingMode();
  ARM_AM::AddrOpc AddSub = (AM == ISD::PRE_INC || AM == ISD::POST_INC)
    ? ARM_AM::add : ARM_AM::sub;
  int Val;
  if (isScaledConstantInRange(N, /*Scale=*/1, 0, 0x1000, Val))
    return false;

  Offset = N;
  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(N.getOpcode());
  unsigned ShAmt = 0;
  if (ShOpcVal != ARM_AM::no_shift) {
    // Check to see if the RHS of the shift is a constant, if not, we can't fold
    // it.
    if (ConstantSDNode *Sh = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
      ShAmt = Sh->getZExtValue();
      if (isShifterOpProfitable(N, ShOpcVal, ShAmt))
        Offset = N.getOperand(0);
      else {
        ShAmt = 0;
        ShOpcVal = ARM_AM::no_shift;
      }
    } else {
      ShOpcVal = ARM_AM::no_shift;
    }
  }

  Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, ShAmt, ShOpcVal),
                                  MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode2OffsetImmPre(SDNode *Op, SDValue N,
                                            SDValue &Offset, SDValue &Opc) {
  unsigned Opcode = Op->getOpcode();
  ISD::MemIndexedMode AM = (Opcode == ISD::LOAD)
    ? cast<LoadSDNode>(Op)->getAddressingMode()
    : cast<StoreSDNode>(Op)->getAddressingMode();
  ARM_AM::AddrOpc AddSub = (AM == ISD::PRE_INC || AM == ISD::POST_INC)
    ? ARM_AM::add : ARM_AM::sub;
  int Val;
  if (isScaledConstantInRange(N, /*Scale=*/1, 0, 0x1000, Val)) { // 12 bits.
    if (AddSub == ARM_AM::sub) Val *= -1;
    Offset = CurDAG->getRegister(0, MVT::i32);
    Opc = CurDAG->getTargetConstant(Val, MVT::i32);
    return true;
  }

  return false;
}


bool ARMDAGToDAGISel::SelectAddrMode2OffsetImm(SDNode *Op, SDValue N,
                                            SDValue &Offset, SDValue &Opc) {
  unsigned Opcode = Op->getOpcode();
  ISD::MemIndexedMode AM = (Opcode == ISD::LOAD)
    ? cast<LoadSDNode>(Op)->getAddressingMode()
    : cast<StoreSDNode>(Op)->getAddressingMode();
  ARM_AM::AddrOpc AddSub = (AM == ISD::PRE_INC || AM == ISD::POST_INC)
    ? ARM_AM::add : ARM_AM::sub;
  int Val;
  if (isScaledConstantInRange(N, /*Scale=*/1, 0, 0x1000, Val)) { // 12 bits.
    Offset = CurDAG->getRegister(0, MVT::i32);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM2Opc(AddSub, Val,
                                                      ARM_AM::no_shift),
                                    MVT::i32);
    return true;
  }

  return false;
}

bool ARMDAGToDAGISel::SelectAddrOffsetNone(SDValue N, SDValue &Base) {
  Base = N;
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode3(SDValue N,
                                      SDValue &Base, SDValue &Offset,
                                      SDValue &Opc) {
  if (N.getOpcode() == ISD::SUB) {
    // X - C  is canonicalize to X + -C, no need to handle it here.
    Base = N.getOperand(0);
    Offset = N.getOperand(1);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(ARM_AM::sub, 0),MVT::i32);
    return true;
  }

  if (!CurDAG->isBaseWithConstantOffset(N)) {
    Base = N;
    if (N.getOpcode() == ISD::FrameIndex) {
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    }
    Offset = CurDAG->getRegister(0, MVT::i32);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(ARM_AM::add, 0),MVT::i32);
    return true;
  }

  // If the RHS is +/- imm8, fold into addr mode.
  int RHSC;
  if (isScaledConstantInRange(N.getOperand(1), /*Scale=*/1,
                              -256 + 1, 256, RHSC)) { // 8 bits.
    Base = N.getOperand(0);
    if (Base.getOpcode() == ISD::FrameIndex) {
      int FI = cast<FrameIndexSDNode>(Base)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    }
    Offset = CurDAG->getRegister(0, MVT::i32);

    ARM_AM::AddrOpc AddSub = ARM_AM::add;
    if (RHSC < 0) {
      AddSub = ARM_AM::sub;
      RHSC = -RHSC;
    }
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(AddSub, RHSC),MVT::i32);
    return true;
  }

  Base = N.getOperand(0);
  Offset = N.getOperand(1);
  Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(ARM_AM::add, 0), MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode3Offset(SDNode *Op, SDValue N,
                                            SDValue &Offset, SDValue &Opc) {
  unsigned Opcode = Op->getOpcode();
  ISD::MemIndexedMode AM = (Opcode == ISD::LOAD)
    ? cast<LoadSDNode>(Op)->getAddressingMode()
    : cast<StoreSDNode>(Op)->getAddressingMode();
  ARM_AM::AddrOpc AddSub = (AM == ISD::PRE_INC || AM == ISD::POST_INC)
    ? ARM_AM::add : ARM_AM::sub;
  int Val;
  if (isScaledConstantInRange(N, /*Scale=*/1, 0, 256, Val)) { // 12 bits.
    Offset = CurDAG->getRegister(0, MVT::i32);
    Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(AddSub, Val), MVT::i32);
    return true;
  }

  Offset = N;
  Opc = CurDAG->getTargetConstant(ARM_AM::getAM3Opc(AddSub, 0), MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode5(SDValue N,
                                      SDValue &Base, SDValue &Offset) {
  if (!CurDAG->isBaseWithConstantOffset(N)) {
    Base = N;
    if (N.getOpcode() == ISD::FrameIndex) {
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    } else if (N.getOpcode() == ARMISD::Wrapper &&
               !(Subtarget->useMovt() &&
                 N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
    }
    Offset = CurDAG->getTargetConstant(ARM_AM::getAM5Opc(ARM_AM::add, 0),
                                       MVT::i32);
    return true;
  }

  // If the RHS is +/- imm8, fold into addr mode.
  int RHSC;
  if (isScaledConstantInRange(N.getOperand(1), /*Scale=*/4,
                              -256 + 1, 256, RHSC)) {
    Base = N.getOperand(0);
    if (Base.getOpcode() == ISD::FrameIndex) {
      int FI = cast<FrameIndexSDNode>(Base)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    }

    ARM_AM::AddrOpc AddSub = ARM_AM::add;
    if (RHSC < 0) {
      AddSub = ARM_AM::sub;
      RHSC = -RHSC;
    }
    Offset = CurDAG->getTargetConstant(ARM_AM::getAM5Opc(AddSub, RHSC),
                                       MVT::i32);
    return true;
  }

  Base = N;
  Offset = CurDAG->getTargetConstant(ARM_AM::getAM5Opc(ARM_AM::add, 0),
                                     MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode6(SDNode *Parent, SDValue N, SDValue &Addr,
                                      SDValue &Align) {
  Addr = N;

  unsigned Alignment = 0;
  if (LSBaseSDNode *LSN = dyn_cast<LSBaseSDNode>(Parent)) {
    // This case occurs only for VLD1-lane/dup and VST1-lane instructions.
    // The maximum alignment is equal to the memory size being referenced.
    unsigned LSNAlign = LSN->getAlignment();
    unsigned MemSize = LSN->getMemoryVT().getSizeInBits() / 8;
    if (LSNAlign >= MemSize && MemSize > 1)
      Alignment = MemSize;
  } else {
    // All other uses of addrmode6 are for intrinsics.  For now just record
    // the raw alignment value; it will be refined later based on the legal
    // alignment operands for the intrinsic.
    Alignment = cast<MemIntrinsicSDNode>(Parent)->getAlignment();
  }

  Align = CurDAG->getTargetConstant(Alignment, MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectAddrMode6Offset(SDNode *Op, SDValue N,
                                            SDValue &Offset) {
  LSBaseSDNode *LdSt = cast<LSBaseSDNode>(Op);
  ISD::MemIndexedMode AM = LdSt->getAddressingMode();
  if (AM != ISD::POST_INC)
    return false;
  Offset = N;
  if (ConstantSDNode *NC = dyn_cast<ConstantSDNode>(N)) {
    if (NC->getZExtValue() * 8 == LdSt->getMemoryVT().getSizeInBits())
      Offset = CurDAG->getRegister(0, MVT::i32);
  }
  return true;
}

bool ARMDAGToDAGISel::SelectAddrModePC(SDValue N,
                                       SDValue &Offset, SDValue &Label) {
  if (N.getOpcode() == ARMISD::PIC_ADD && N.hasOneUse()) {
    Offset = N.getOperand(0);
    SDValue N1 = N.getOperand(1);
    Label = CurDAG->getTargetConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                                      MVT::i32);
    return true;
  }

  return false;
}


//===----------------------------------------------------------------------===//
//                         Thumb Addressing Modes
//===----------------------------------------------------------------------===//

bool ARMDAGToDAGISel::SelectThumbAddrModeRR(SDValue N,
                                            SDValue &Base, SDValue &Offset){
  if (N.getOpcode() != ISD::ADD && !CurDAG->isBaseWithConstantOffset(N)) {
    ConstantSDNode *NC = dyn_cast<ConstantSDNode>(N);
    if (!NC || !NC->isNullValue())
      return false;

    Base = Offset = N;
    return true;
  }

  Base = N.getOperand(0);
  Offset = N.getOperand(1);
  return true;
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeRI(SDValue N, SDValue &Base,
                                       SDValue &Offset, unsigned Scale) {
  if (Scale == 4) {
    SDValue TmpBase, TmpOffImm;
    if (SelectThumbAddrModeSP(N, TmpBase, TmpOffImm))
      return false;  // We want to select tLDRspi / tSTRspi instead.

    if (N.getOpcode() == ARMISD::Wrapper &&
        N.getOperand(0).getOpcode() == ISD::TargetConstantPool)
      return false;  // We want to select tLDRpci instead.
  }

  if (!CurDAG->isBaseWithConstantOffset(N))
    return false;

  // Thumb does not have [sp, r] address mode.
  RegisterSDNode *LHSR = dyn_cast<RegisterSDNode>(N.getOperand(0));
  RegisterSDNode *RHSR = dyn_cast<RegisterSDNode>(N.getOperand(1));
  if ((LHSR && LHSR->getReg() == ARM::SP) ||
      (RHSR && RHSR->getReg() == ARM::SP))
    return false;

  // FIXME: Why do we explicitly check for a match here and then return false?
  // Presumably to allow something else to match, but shouldn't this be
  // documented?
  int RHSC;
  if (isScaledConstantInRange(N.getOperand(1), Scale, 0, 32, RHSC))
    return false;

  Base = N.getOperand(0);
  Offset = N.getOperand(1);
  return true;
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeRI5S1(SDValue N,
                                          SDValue &Base,
                                          SDValue &Offset) {
  return SelectThumbAddrModeRI(N, Base, Offset, 1);
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeRI5S2(SDValue N,
                                          SDValue &Base,
                                          SDValue &Offset) {
  return SelectThumbAddrModeRI(N, Base, Offset, 2);
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeRI5S4(SDValue N,
                                          SDValue &Base,
                                          SDValue &Offset) {
  return SelectThumbAddrModeRI(N, Base, Offset, 4);
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeImm5S(SDValue N, unsigned Scale,
                                          SDValue &Base, SDValue &OffImm) {
  if (Scale == 4) {
    SDValue TmpBase, TmpOffImm;
    if (SelectThumbAddrModeSP(N, TmpBase, TmpOffImm))
      return false;  // We want to select tLDRspi / tSTRspi instead.

    if (N.getOpcode() == ARMISD::Wrapper &&
        N.getOperand(0).getOpcode() == ISD::TargetConstantPool)
      return false;  // We want to select tLDRpci instead.
  }

  if (!CurDAG->isBaseWithConstantOffset(N)) {
    if (N.getOpcode() == ARMISD::Wrapper &&
        !(Subtarget->useMovt() &&
          N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
    } else {
      Base = N;
    }

    OffImm = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  RegisterSDNode *LHSR = dyn_cast<RegisterSDNode>(N.getOperand(0));
  RegisterSDNode *RHSR = dyn_cast<RegisterSDNode>(N.getOperand(1));
  if ((LHSR && LHSR->getReg() == ARM::SP) ||
      (RHSR && RHSR->getReg() == ARM::SP)) {
    ConstantSDNode *LHS = dyn_cast<ConstantSDNode>(N.getOperand(0));
    ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1));
    unsigned LHSC = LHS ? LHS->getZExtValue() : 0;
    unsigned RHSC = RHS ? RHS->getZExtValue() : 0;

    // Thumb does not have [sp, #imm5] address mode for non-zero imm5.
    if (LHSC != 0 || RHSC != 0) return false;

    Base = N;
    OffImm = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  // If the RHS is + imm5 * scale, fold into addr mode.
  int RHSC;
  if (isScaledConstantInRange(N.getOperand(1), Scale, 0, 32, RHSC)) {
    Base = N.getOperand(0);
    OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
    return true;
  }

  Base = N.getOperand(0);
  OffImm = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeImm5S4(SDValue N, SDValue &Base,
                                           SDValue &OffImm) {
  return SelectThumbAddrModeImm5S(N, 4, Base, OffImm);
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeImm5S2(SDValue N, SDValue &Base,
                                           SDValue &OffImm) {
  return SelectThumbAddrModeImm5S(N, 2, Base, OffImm);
}

bool
ARMDAGToDAGISel::SelectThumbAddrModeImm5S1(SDValue N, SDValue &Base,
                                           SDValue &OffImm) {
  return SelectThumbAddrModeImm5S(N, 1, Base, OffImm);
}

bool ARMDAGToDAGISel::SelectThumbAddrModeSP(SDValue N,
                                            SDValue &Base, SDValue &OffImm) {
  if (N.getOpcode() == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    OffImm = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  if (!CurDAG->isBaseWithConstantOffset(N))
    return false;

  RegisterSDNode *LHSR = dyn_cast<RegisterSDNode>(N.getOperand(0));
  if (N.getOperand(0).getOpcode() == ISD::FrameIndex ||
      (LHSR && LHSR->getReg() == ARM::SP)) {
    // If the RHS is + imm8 * scale, fold into addr mode.
    int RHSC;
    if (isScaledConstantInRange(N.getOperand(1), /*Scale=*/4, 0, 256, RHSC)) {
      Base = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      }
      OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
      return true;
    }
  }

  return false;
}


//===----------------------------------------------------------------------===//
//                        Thumb 2 Addressing Modes
//===----------------------------------------------------------------------===//


bool ARMDAGToDAGISel::SelectT2ShifterOperandReg(SDValue N, SDValue &BaseReg,
                                                SDValue &Opc) {
  if (DisableShifterOp)
    return false;

  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(N.getOpcode());

  // Don't match base register only case. That is matched to a separate
  // lower complexity pattern with explicit register operand.
  if (ShOpcVal == ARM_AM::no_shift) return false;

  BaseReg = N.getOperand(0);
  unsigned ShImmVal = 0;
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    ShImmVal = RHS->getZExtValue() & 31;
    Opc = getI32Imm(ARM_AM::getSORegOpc(ShOpcVal, ShImmVal));
    return true;
  }

  return false;
}

bool ARMDAGToDAGISel::SelectT2AddrModeImm12(SDValue N,
                                            SDValue &Base, SDValue &OffImm) {
  // Match simple R + imm12 operands.

  // Base only.
  if (N.getOpcode() != ISD::ADD && N.getOpcode() != ISD::SUB &&
      !CurDAG->isBaseWithConstantOffset(N)) {
    if (N.getOpcode() == ISD::FrameIndex) {
      // Match frame index.
      int FI = cast<FrameIndexSDNode>(N)->getIndex();
      Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
      return true;
    }

    if (N.getOpcode() == ARMISD::Wrapper &&
               !(Subtarget->useMovt() &&
                 N.getOperand(0).getOpcode() == ISD::TargetGlobalAddress)) {
      Base = N.getOperand(0);
      if (Base.getOpcode() == ISD::TargetConstantPool)
        return false;  // We want to select t2LDRpci instead.
    } else
      Base = N;
    OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }

  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    if (SelectT2AddrModeImm8(N, Base, OffImm))
      // Let t2LDRi8 handle (R - imm8).
      return false;

    int RHSC = (int)RHS->getZExtValue();
    if (N.getOpcode() == ISD::SUB)
      RHSC = -RHSC;

    if (RHSC >= 0 && RHSC < 0x1000) { // 12 bits (unsigned)
      Base   = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      }
      OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
      return true;
    }
  }

  // Base only.
  Base = N;
  OffImm  = CurDAG->getTargetConstant(0, MVT::i32);
  return true;
}

bool ARMDAGToDAGISel::SelectT2AddrModeImm8(SDValue N,
                                           SDValue &Base, SDValue &OffImm) {
  // Match simple R - imm8 operands.
  if (N.getOpcode() != ISD::ADD && N.getOpcode() != ISD::SUB &&
      !CurDAG->isBaseWithConstantOffset(N))
    return false;

  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int RHSC = (int)RHS->getSExtValue();
    if (N.getOpcode() == ISD::SUB)
      RHSC = -RHSC;

    if ((RHSC >= -255) && (RHSC < 0)) { // 8 bits (always negative)
      Base = N.getOperand(0);
      if (Base.getOpcode() == ISD::FrameIndex) {
        int FI = cast<FrameIndexSDNode>(Base)->getIndex();
        Base = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
      }
      OffImm = CurDAG->getTargetConstant(RHSC, MVT::i32);
      return true;
    }
  }

  return false;
}

bool ARMDAGToDAGISel::SelectT2AddrModeImm8Offset(SDNode *Op, SDValue N,
                                                 SDValue &OffImm){
  unsigned Opcode = Op->getOpcode();
  ISD::MemIndexedMode AM = (Opcode == ISD::LOAD)
    ? cast<LoadSDNode>(Op)->getAddressingMode()
    : cast<StoreSDNode>(Op)->getAddressingMode();
  int RHSC;
  if (isScaledConstantInRange(N, /*Scale=*/1, 0, 0x100, RHSC)) { // 8 bits.
    OffImm = ((AM == ISD::PRE_INC) || (AM == ISD::POST_INC))
      ? CurDAG->getTargetConstant(RHSC, MVT::i32)
      : CurDAG->getTargetConstant(-RHSC, MVT::i32);
    return true;
  }

  return false;
}

bool ARMDAGToDAGISel::SelectT2AddrModeSoReg(SDValue N,
                                            SDValue &Base,
                                            SDValue &OffReg, SDValue &ShImm) {
  // (R - imm8) should be handled by t2LDRi8. The rest are handled by t2LDRi12.
  if (N.getOpcode() != ISD::ADD && !CurDAG->isBaseWithConstantOffset(N))
    return false;

  // Leave (R + imm12) for t2LDRi12, (R - imm8) for t2LDRi8.
  if (ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N.getOperand(1))) {
    int RHSC = (int)RHS->getZExtValue();
    if (RHSC >= 0 && RHSC < 0x1000) // 12 bits (unsigned)
      return false;
    else if (RHSC < 0 && RHSC >= -255) // 8 bits
      return false;
  }

  // Look for (R + R) or (R + (R << [1,2,3])).
  unsigned ShAmt = 0;
  Base   = N.getOperand(0);
  OffReg = N.getOperand(1);

  // Swap if it is ((R << c) + R).
  ARM_AM::ShiftOpc ShOpcVal = ARM_AM::getShiftOpcForNode(OffReg.getOpcode());
  if (ShOpcVal != ARM_AM::lsl) {
    ShOpcVal = ARM_AM::getShiftOpcForNode(Base.getOpcode());
    if (ShOpcVal == ARM_AM::lsl)
      std::swap(Base, OffReg);
  }

  if (ShOpcVal == ARM_AM::lsl) {
    // Check to see if the RHS of the shift is a constant, if not, we can't fold
    // it.
    if (ConstantSDNode *Sh = dyn_cast<ConstantSDNode>(OffReg.getOperand(1))) {
      ShAmt = Sh->getZExtValue();
      if (ShAmt < 4 && isShifterOpProfitable(OffReg, ShOpcVal, ShAmt))
        OffReg = OffReg.getOperand(0);
      else {
        ShAmt = 0;
        ShOpcVal = ARM_AM::no_shift;
      }
    } else {
      ShOpcVal = ARM_AM::no_shift;
    }
  }

  ShImm = CurDAG->getTargetConstant(ShAmt, MVT::i32);

  return true;
}

//===--------------------------------------------------------------------===//

/// getAL - Returns a ARMCC::AL immediate node.
static inline SDValue getAL(SelectionDAG *CurDAG) {
  return CurDAG->getTargetConstant((uint64_t)ARMCC::AL, MVT::i32);
}

SDNode *ARMDAGToDAGISel::SelectARMIndexedLoad(SDNode *N) {
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::MemIndexedMode AM = LD->getAddressingMode();
  if (AM == ISD::UNINDEXED)
    return NULL;

  EVT LoadedVT = LD->getMemoryVT();
  SDValue Offset, AMOpc;
  bool isPre = (AM == ISD::PRE_INC) || (AM == ISD::PRE_DEC);
  unsigned Opcode = 0;
  bool Match = false;
  if (LoadedVT == MVT::i32 && isPre &&
      SelectAddrMode2OffsetImmPre(N, LD->getOffset(), Offset, AMOpc)) {
    Opcode = ARM::LDR_PRE_IMM;
    Match = true;
  } else if (LoadedVT == MVT::i32 && !isPre &&
      SelectAddrMode2OffsetImm(N, LD->getOffset(), Offset, AMOpc)) {
    Opcode = ARM::LDR_POST_IMM;
    Match = true;
  } else if (LoadedVT == MVT::i32 &&
      SelectAddrMode2OffsetReg(N, LD->getOffset(), Offset, AMOpc)) {
    Opcode = isPre ? ARM::LDR_PRE_REG : ARM::LDR_POST_REG;
    Match = true;

  } else if (LoadedVT == MVT::i16 &&
             SelectAddrMode3Offset(N, LD->getOffset(), Offset, AMOpc)) {
    Match = true;
    Opcode = (LD->getExtensionType() == ISD::SEXTLOAD)
      ? (isPre ? ARM::LDRSH_PRE : ARM::LDRSH_POST)
      : (isPre ? ARM::LDRH_PRE : ARM::LDRH_POST);
  } else if (LoadedVT == MVT::i8 || LoadedVT == MVT::i1) {
    if (LD->getExtensionType() == ISD::SEXTLOAD) {
      if (SelectAddrMode3Offset(N, LD->getOffset(), Offset, AMOpc)) {
        Match = true;
        Opcode = isPre ? ARM::LDRSB_PRE : ARM::LDRSB_POST;
      }
    } else {
      if (isPre &&
          SelectAddrMode2OffsetImmPre(N, LD->getOffset(), Offset, AMOpc)) {
        Match = true;
        Opcode = ARM::LDRB_PRE_IMM;
      } else if (!isPre &&
                  SelectAddrMode2OffsetImm(N, LD->getOffset(), Offset, AMOpc)) {
        Match = true;
        Opcode = ARM::LDRB_POST_IMM;
      } else if (SelectAddrMode2OffsetReg(N, LD->getOffset(), Offset, AMOpc)) {
        Match = true;
        Opcode = isPre ? ARM::LDRB_PRE_REG : ARM::LDRB_POST_REG;
      }
    }
  }

  if (Match) {
    if (Opcode == ARM::LDR_PRE_IMM || Opcode == ARM::LDRB_PRE_IMM) {
      SDValue Chain = LD->getChain();
      SDValue Base = LD->getBasePtr();
      SDValue Ops[]= { Base, AMOpc, getAL(CurDAG),
                       CurDAG->getRegister(0, MVT::i32), Chain };
      return CurDAG->getMachineNode(Opcode, N->getDebugLoc(), MVT::i32,
                                    MVT::i32, MVT::Other, Ops, 5);
    } else {
      SDValue Chain = LD->getChain();
      SDValue Base = LD->getBasePtr();
      SDValue Ops[]= { Base, Offset, AMOpc, getAL(CurDAG),
                       CurDAG->getRegister(0, MVT::i32), Chain };
      return CurDAG->getMachineNode(Opcode, N->getDebugLoc(), MVT::i32,
                                    MVT::i32, MVT::Other, Ops, 6);
    }
  }

  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectT2IndexedLoad(SDNode *N) {
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::MemIndexedMode AM = LD->getAddressingMode();
  if (AM == ISD::UNINDEXED)
    return NULL;

  EVT LoadedVT = LD->getMemoryVT();
  bool isSExtLd = LD->getExtensionType() == ISD::SEXTLOAD;
  SDValue Offset;
  bool isPre = (AM == ISD::PRE_INC) || (AM == ISD::PRE_DEC);
  unsigned Opcode = 0;
  bool Match = false;
  if (SelectT2AddrModeImm8Offset(N, LD->getOffset(), Offset)) {
    switch (LoadedVT.getSimpleVT().SimpleTy) {
    case MVT::i32:
      Opcode = isPre ? ARM::t2LDR_PRE : ARM::t2LDR_POST;
      break;
    case MVT::i16:
      if (isSExtLd)
        Opcode = isPre ? ARM::t2LDRSH_PRE : ARM::t2LDRSH_POST;
      else
        Opcode = isPre ? ARM::t2LDRH_PRE : ARM::t2LDRH_POST;
      break;
    case MVT::i8:
    case MVT::i1:
      if (isSExtLd)
        Opcode = isPre ? ARM::t2LDRSB_PRE : ARM::t2LDRSB_POST;
      else
        Opcode = isPre ? ARM::t2LDRB_PRE : ARM::t2LDRB_POST;
      break;
    default:
      return NULL;
    }
    Match = true;
  }

  if (Match) {
    SDValue Chain = LD->getChain();
    SDValue Base = LD->getBasePtr();
    SDValue Ops[]= { Base, Offset, getAL(CurDAG),
                     CurDAG->getRegister(0, MVT::i32), Chain };
    return CurDAG->getMachineNode(Opcode, N->getDebugLoc(), MVT::i32, MVT::i32,
                                  MVT::Other, Ops, 5);
  }

  return NULL;
}

/// PairSRegs - Form a D register from a pair of S registers.
///
SDNode *ARMDAGToDAGISel::PairSRegs(EVT VT, SDValue V0, SDValue V1) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue RegClass =
    CurDAG->getTargetConstant(ARM::DPR_VFP2RegClassID, MVT::i32);
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::ssub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::ssub_1, MVT::i32);
  const SDValue Ops[] = { RegClass, V0, SubReg0, V1, SubReg1 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 5);
}

/// PairDRegs - Form a quad register from a pair of D registers.
///
SDNode *ARMDAGToDAGISel::PairDRegs(EVT VT, SDValue V0, SDValue V1) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue RegClass = CurDAG->getTargetConstant(ARM::QPRRegClassID, MVT::i32);
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::dsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::dsub_1, MVT::i32);
  const SDValue Ops[] = { RegClass, V0, SubReg0, V1, SubReg1 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 5);
}

/// PairQRegs - Form 4 consecutive D registers from a pair of Q registers.
///
SDNode *ARMDAGToDAGISel::PairQRegs(EVT VT, SDValue V0, SDValue V1) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue RegClass = CurDAG->getTargetConstant(ARM::QQPRRegClassID, MVT::i32);
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::qsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::qsub_1, MVT::i32);
  const SDValue Ops[] = { RegClass, V0, SubReg0, V1, SubReg1 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 5);
}

/// QuadSRegs - Form 4 consecutive S registers.
///
SDNode *ARMDAGToDAGISel::QuadSRegs(EVT VT, SDValue V0, SDValue V1,
                                   SDValue V2, SDValue V3) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue RegClass =
    CurDAG->getTargetConstant(ARM::QPR_VFP2RegClassID, MVT::i32);
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::ssub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::ssub_1, MVT::i32);
  SDValue SubReg2 = CurDAG->getTargetConstant(ARM::ssub_2, MVT::i32);
  SDValue SubReg3 = CurDAG->getTargetConstant(ARM::ssub_3, MVT::i32);
  const SDValue Ops[] = { RegClass, V0, SubReg0, V1, SubReg1,
                                    V2, SubReg2, V3, SubReg3 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 9);
}

/// QuadDRegs - Form 4 consecutive D registers.
///
SDNode *ARMDAGToDAGISel::QuadDRegs(EVT VT, SDValue V0, SDValue V1,
                                   SDValue V2, SDValue V3) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue RegClass = CurDAG->getTargetConstant(ARM::QQPRRegClassID, MVT::i32);
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::dsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::dsub_1, MVT::i32);
  SDValue SubReg2 = CurDAG->getTargetConstant(ARM::dsub_2, MVT::i32);
  SDValue SubReg3 = CurDAG->getTargetConstant(ARM::dsub_3, MVT::i32);
  const SDValue Ops[] = { RegClass, V0, SubReg0, V1, SubReg1,
                                    V2, SubReg2, V3, SubReg3 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 9);
}

/// QuadQRegs - Form 4 consecutive Q registers.
///
SDNode *ARMDAGToDAGISel::QuadQRegs(EVT VT, SDValue V0, SDValue V1,
                                   SDValue V2, SDValue V3) {
  DebugLoc dl = V0.getNode()->getDebugLoc();
  SDValue RegClass = CurDAG->getTargetConstant(ARM::QQQQPRRegClassID, MVT::i32);
  SDValue SubReg0 = CurDAG->getTargetConstant(ARM::qsub_0, MVT::i32);
  SDValue SubReg1 = CurDAG->getTargetConstant(ARM::qsub_1, MVT::i32);
  SDValue SubReg2 = CurDAG->getTargetConstant(ARM::qsub_2, MVT::i32);
  SDValue SubReg3 = CurDAG->getTargetConstant(ARM::qsub_3, MVT::i32);
  const SDValue Ops[] = { RegClass, V0, SubReg0, V1, SubReg1,
                                    V2, SubReg2, V3, SubReg3 };
  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, dl, VT, Ops, 9);
}

/// GetVLDSTAlign - Get the alignment (in bytes) for the alignment operand
/// of a NEON VLD or VST instruction.  The supported values depend on the
/// number of registers being loaded.
SDValue ARMDAGToDAGISel::GetVLDSTAlign(SDValue Align, unsigned NumVecs,
                                       bool is64BitVector) {
  unsigned NumRegs = NumVecs;
  if (!is64BitVector && NumVecs < 3)
    NumRegs *= 2;

  unsigned Alignment = cast<ConstantSDNode>(Align)->getZExtValue();
  if (Alignment >= 32 && NumRegs == 4)
    Alignment = 32;
  else if (Alignment >= 16 && (NumRegs == 2 || NumRegs == 4))
    Alignment = 16;
  else if (Alignment >= 8)
    Alignment = 8;
  else
    Alignment = 0;

  return CurDAG->getTargetConstant(Alignment, MVT::i32);
}

// Get the register stride update opcode of a VLD/VST instruction that
// is otherwise equivalent to the given fixed stride updating instruction.
static unsigned getVLDSTRegisterUpdateOpcode(unsigned Opc) {
  switch (Opc) {
  default: break;
  case ARM::VLD1d8wb_fixed: return ARM::VLD1d8wb_register;
  case ARM::VLD1d16wb_fixed: return ARM::VLD1d16wb_register;
  case ARM::VLD1d32wb_fixed: return ARM::VLD1d32wb_register;
  case ARM::VLD1d64wb_fixed: return ARM::VLD1d64wb_register;
  case ARM::VLD1q8wb_fixed: return ARM::VLD1q8wb_register;
  case ARM::VLD1q16wb_fixed: return ARM::VLD1q16wb_register;
  case ARM::VLD1q32wb_fixed: return ARM::VLD1q32wb_register;
  case ARM::VLD1q64wb_fixed: return ARM::VLD1q64wb_register;

  case ARM::VST1d8wb_fixed: return ARM::VST1d8wb_register;
  case ARM::VST1d16wb_fixed: return ARM::VST1d16wb_register;
  case ARM::VST1d32wb_fixed: return ARM::VST1d32wb_register;
  case ARM::VST1d64wb_fixed: return ARM::VST1d64wb_register;
  case ARM::VST1q8wb_fixed: return ARM::VST1q8wb_register;
  case ARM::VST1q16wb_fixed: return ARM::VST1q16wb_register;
  case ARM::VST1q32wb_fixed: return ARM::VST1q32wb_register;
  case ARM::VST1q64wb_fixed: return ARM::VST1q64wb_register;
  case ARM::VST1d64TPseudoWB_fixed: return ARM::VST1d64TPseudoWB_register;
  case ARM::VST1d64QPseudoWB_fixed: return ARM::VST1d64QPseudoWB_register;

  case ARM::VLD2d8wb_fixed: return ARM::VLD2d8wb_register;
  case ARM::VLD2d16wb_fixed: return ARM::VLD2d16wb_register;
  case ARM::VLD2d32wb_fixed: return ARM::VLD2d32wb_register;
  case ARM::VLD2q8PseudoWB_fixed: return ARM::VLD2q8PseudoWB_register;
  case ARM::VLD2q16PseudoWB_fixed: return ARM::VLD2q16PseudoWB_register;
  case ARM::VLD2q32PseudoWB_fixed: return ARM::VLD2q32PseudoWB_register;

  case ARM::VST2d8wb_fixed: return ARM::VST2d8wb_register;
  case ARM::VST2d16wb_fixed: return ARM::VST2d16wb_register;
  case ARM::VST2d32wb_fixed: return ARM::VST2d32wb_register;
  case ARM::VST2q8PseudoWB_fixed: return ARM::VST2q8PseudoWB_register;
  case ARM::VST2q16PseudoWB_fixed: return ARM::VST2q16PseudoWB_register;
  case ARM::VST2q32PseudoWB_fixed: return ARM::VST2q32PseudoWB_register;

  case ARM::VLD2DUPd8wb_fixed: return ARM::VLD2DUPd8wb_register;
  case ARM::VLD2DUPd16wb_fixed: return ARM::VLD2DUPd16wb_register;
  case ARM::VLD2DUPd32wb_fixed: return ARM::VLD2DUPd32wb_register;
  }
  return Opc; // If not one we handle, return it unchanged.
}

SDNode *ARMDAGToDAGISel::SelectVLD(SDNode *N, bool isUpdating, unsigned NumVecs,
                                   const uint16_t *DOpcodes,
                                   const uint16_t *QOpcodes0,
                                   const uint16_t *QOpcodes1) {
  assert(NumVecs >= 1 && NumVecs <= 4 && "VLD NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();

  SDValue MemAddr, Align;
  unsigned AddrOpIdx = isUpdating ? 1 : 2;
  if (!SelectAddrMode6(N, N->getOperand(AddrOpIdx), MemAddr, Align))
    return NULL;

  SDValue Chain = N->getOperand(0);
  EVT VT = N->getValueType(0);
  bool is64BitVector = VT.is64BitVector();
  Align = GetVLDSTAlign(Align, NumVecs, is64BitVector);

  unsigned OpcodeIndex;
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("unhandled vld type");
    // Double-register operations:
  case MVT::v8i8:  OpcodeIndex = 0; break;
  case MVT::v4i16: OpcodeIndex = 1; break;
  case MVT::v2f32:
  case MVT::v2i32: OpcodeIndex = 2; break;
  case MVT::v1i64: OpcodeIndex = 3; break;
    // Quad-register operations:
  case MVT::v16i8: OpcodeIndex = 0; break;
  case MVT::v8i16: OpcodeIndex = 1; break;
  case MVT::v4f32:
  case MVT::v4i32: OpcodeIndex = 2; break;
  case MVT::v2i64: OpcodeIndex = 3;
    assert(NumVecs == 1 && "v2i64 type only supported for VLD1");
    break;
  }

  EVT ResTy;
  if (NumVecs == 1)
    ResTy = VT;
  else {
    unsigned ResTyElts = (NumVecs == 3) ? 4 : NumVecs;
    if (!is64BitVector)
      ResTyElts *= 2;
    ResTy = EVT::getVectorVT(*CurDAG->getContext(), MVT::i64, ResTyElts);
  }
  std::vector<EVT> ResTys;
  ResTys.push_back(ResTy);
  if (isUpdating)
    ResTys.push_back(MVT::i32);
  ResTys.push_back(MVT::Other);

  SDValue Pred = getAL(CurDAG);
  SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
  SDNode *VLd;
  SmallVector<SDValue, 7> Ops;

  // Double registers and VLD1/VLD2 quad registers are directly supported.
  if (is64BitVector || NumVecs <= 2) {
    unsigned Opc = (is64BitVector ? DOpcodes[OpcodeIndex] :
                    QOpcodes0[OpcodeIndex]);
    Ops.push_back(MemAddr);
    Ops.push_back(Align);
    if (isUpdating) {
      SDValue Inc = N->getOperand(AddrOpIdx + 1);
      // FIXME: VLD1/VLD2 fixed increment doesn't need Reg0. Remove the reg0
      // case entirely when the rest are updated to that form, too.
      if ((NumVecs == 1 || NumVecs == 2) && !isa<ConstantSDNode>(Inc.getNode()))
        Opc = getVLDSTRegisterUpdateOpcode(Opc);
      // We use a VLD1 for v1i64 even if the pseudo says vld2/3/4, so
      // check for that explicitly too. Horribly hacky, but temporary.
      if ((NumVecs != 1 && NumVecs != 2 && Opc != ARM::VLD1q64wb_fixed) ||
          !isa<ConstantSDNode>(Inc.getNode()))
        Ops.push_back(isa<ConstantSDNode>(Inc.getNode()) ? Reg0 : Inc);
    }
    Ops.push_back(Pred);
    Ops.push_back(Reg0);
    Ops.push_back(Chain);
    VLd = CurDAG->getMachineNode(Opc, dl, ResTys, Ops.data(), Ops.size());

  } else {
    // Otherwise, quad registers are loaded with two separate instructions,
    // where one loads the even registers and the other loads the odd registers.
    EVT AddrTy = MemAddr.getValueType();

    // Load the even subregs.  This is always an updating load, so that it
    // provides the address to the second load for the odd subregs.
    SDValue ImplDef =
      SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, dl, ResTy), 0);
    const SDValue OpsA[] = { MemAddr, Align, Reg0, ImplDef, Pred, Reg0, Chain };
    SDNode *VLdA = CurDAG->getMachineNode(QOpcodes0[OpcodeIndex], dl,
                                          ResTy, AddrTy, MVT::Other, OpsA, 7);
    Chain = SDValue(VLdA, 2);

    // Load the odd subregs.
    Ops.push_back(SDValue(VLdA, 1));
    Ops.push_back(Align);
    if (isUpdating) {
      SDValue Inc = N->getOperand(AddrOpIdx + 1);
      assert(isa<ConstantSDNode>(Inc.getNode()) &&
             "only constant post-increment update allowed for VLD3/4");
      (void)Inc;
      Ops.push_back(Reg0);
    }
    Ops.push_back(SDValue(VLdA, 0));
    Ops.push_back(Pred);
    Ops.push_back(Reg0);
    Ops.push_back(Chain);
    VLd = CurDAG->getMachineNode(QOpcodes1[OpcodeIndex], dl, ResTys,
                                 Ops.data(), Ops.size());
  }

  // Transfer memoperands.
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();
  cast<MachineSDNode>(VLd)->setMemRefs(MemOp, MemOp + 1);

  if (NumVecs == 1)
    return VLd;

  // Extract out the subregisters.
  SDValue SuperReg = SDValue(VLd, 0);
  assert(ARM::dsub_7 == ARM::dsub_0+7 &&
         ARM::qsub_3 == ARM::qsub_0+3 && "Unexpected subreg numbering");
  unsigned Sub0 = (is64BitVector ? ARM::dsub_0 : ARM::qsub_0);
  for (unsigned Vec = 0; Vec < NumVecs; ++Vec)
    ReplaceUses(SDValue(N, Vec),
                CurDAG->getTargetExtractSubreg(Sub0 + Vec, dl, VT, SuperReg));
  ReplaceUses(SDValue(N, NumVecs), SDValue(VLd, 1));
  if (isUpdating)
    ReplaceUses(SDValue(N, NumVecs + 1), SDValue(VLd, 2));
  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectVST(SDNode *N, bool isUpdating, unsigned NumVecs,
                                   const uint16_t *DOpcodes,
                                   const uint16_t *QOpcodes0,
                                   const uint16_t *QOpcodes1) {
  assert(NumVecs >= 1 && NumVecs <= 4 && "VST NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();

  SDValue MemAddr, Align;
  unsigned AddrOpIdx = isUpdating ? 1 : 2;
  unsigned Vec0Idx = 3; // AddrOpIdx + (isUpdating ? 2 : 1)
  if (!SelectAddrMode6(N, N->getOperand(AddrOpIdx), MemAddr, Align))
    return NULL;

  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();

  SDValue Chain = N->getOperand(0);
  EVT VT = N->getOperand(Vec0Idx).getValueType();
  bool is64BitVector = VT.is64BitVector();
  Align = GetVLDSTAlign(Align, NumVecs, is64BitVector);

  unsigned OpcodeIndex;
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("unhandled vst type");
    // Double-register operations:
  case MVT::v8i8:  OpcodeIndex = 0; break;
  case MVT::v4i16: OpcodeIndex = 1; break;
  case MVT::v2f32:
  case MVT::v2i32: OpcodeIndex = 2; break;
  case MVT::v1i64: OpcodeIndex = 3; break;
    // Quad-register operations:
  case MVT::v16i8: OpcodeIndex = 0; break;
  case MVT::v8i16: OpcodeIndex = 1; break;
  case MVT::v4f32:
  case MVT::v4i32: OpcodeIndex = 2; break;
  case MVT::v2i64: OpcodeIndex = 3;
    assert(NumVecs == 1 && "v2i64 type only supported for VST1");
    break;
  }

  std::vector<EVT> ResTys;
  if (isUpdating)
    ResTys.push_back(MVT::i32);
  ResTys.push_back(MVT::Other);

  SDValue Pred = getAL(CurDAG);
  SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
  SmallVector<SDValue, 7> Ops;

  // Double registers and VST1/VST2 quad registers are directly supported.
  if (is64BitVector || NumVecs <= 2) {
    SDValue SrcReg;
    if (NumVecs == 1) {
      SrcReg = N->getOperand(Vec0Idx);
    } else if (is64BitVector) {
      // Form a REG_SEQUENCE to force register allocation.
      SDValue V0 = N->getOperand(Vec0Idx + 0);
      SDValue V1 = N->getOperand(Vec0Idx + 1);
      if (NumVecs == 2)
        SrcReg = SDValue(PairDRegs(MVT::v2i64, V0, V1), 0);
      else {
        SDValue V2 = N->getOperand(Vec0Idx + 2);
        // If it's a vst3, form a quad D-register and leave the last part as
        // an undef.
        SDValue V3 = (NumVecs == 3)
          ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF,dl,VT), 0)
          : N->getOperand(Vec0Idx + 3);
        SrcReg = SDValue(QuadDRegs(MVT::v4i64, V0, V1, V2, V3), 0);
      }
    } else {
      // Form a QQ register.
      SDValue Q0 = N->getOperand(Vec0Idx);
      SDValue Q1 = N->getOperand(Vec0Idx + 1);
      SrcReg = SDValue(PairQRegs(MVT::v4i64, Q0, Q1), 0);
    }

    unsigned Opc = (is64BitVector ? DOpcodes[OpcodeIndex] :
                    QOpcodes0[OpcodeIndex]);
    Ops.push_back(MemAddr);
    Ops.push_back(Align);
    if (isUpdating) {
      SDValue Inc = N->getOperand(AddrOpIdx + 1);
      // FIXME: VST1/VST2 fixed increment doesn't need Reg0. Remove the reg0
      // case entirely when the rest are updated to that form, too.
      if (NumVecs <= 2 && !isa<ConstantSDNode>(Inc.getNode()))
        Opc = getVLDSTRegisterUpdateOpcode(Opc);
      // We use a VST1 for v1i64 even if the pseudo says vld2/3/4, so
      // check for that explicitly too. Horribly hacky, but temporary.
      if ((NumVecs > 2 && Opc != ARM::VST1q64wb_fixed) ||
          !isa<ConstantSDNode>(Inc.getNode()))
        Ops.push_back(isa<ConstantSDNode>(Inc.getNode()) ? Reg0 : Inc);
    }
    Ops.push_back(SrcReg);
    Ops.push_back(Pred);
    Ops.push_back(Reg0);
    Ops.push_back(Chain);
    SDNode *VSt =
      CurDAG->getMachineNode(Opc, dl, ResTys, Ops.data(), Ops.size());

    // Transfer memoperands.
    cast<MachineSDNode>(VSt)->setMemRefs(MemOp, MemOp + 1);

    return VSt;
  }

  // Otherwise, quad registers are stored with two separate instructions,
  // where one stores the even registers and the other stores the odd registers.

  // Form the QQQQ REG_SEQUENCE.
  SDValue V0 = N->getOperand(Vec0Idx + 0);
  SDValue V1 = N->getOperand(Vec0Idx + 1);
  SDValue V2 = N->getOperand(Vec0Idx + 2);
  SDValue V3 = (NumVecs == 3)
    ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, dl, VT), 0)
    : N->getOperand(Vec0Idx + 3);
  SDValue RegSeq = SDValue(QuadQRegs(MVT::v8i64, V0, V1, V2, V3), 0);

  // Store the even D registers.  This is always an updating store, so that it
  // provides the address to the second store for the odd subregs.
  const SDValue OpsA[] = { MemAddr, Align, Reg0, RegSeq, Pred, Reg0, Chain };
  SDNode *VStA = CurDAG->getMachineNode(QOpcodes0[OpcodeIndex], dl,
                                        MemAddr.getValueType(),
                                        MVT::Other, OpsA, 7);
  cast<MachineSDNode>(VStA)->setMemRefs(MemOp, MemOp + 1);
  Chain = SDValue(VStA, 1);

  // Store the odd D registers.
  Ops.push_back(SDValue(VStA, 0));
  Ops.push_back(Align);
  if (isUpdating) {
    SDValue Inc = N->getOperand(AddrOpIdx + 1);
    assert(isa<ConstantSDNode>(Inc.getNode()) &&
           "only constant post-increment update allowed for VST3/4");
    (void)Inc;
    Ops.push_back(Reg0);
  }
  Ops.push_back(RegSeq);
  Ops.push_back(Pred);
  Ops.push_back(Reg0);
  Ops.push_back(Chain);
  SDNode *VStB = CurDAG->getMachineNode(QOpcodes1[OpcodeIndex], dl, ResTys,
                                        Ops.data(), Ops.size());
  cast<MachineSDNode>(VStB)->setMemRefs(MemOp, MemOp + 1);
  return VStB;
}

SDNode *ARMDAGToDAGISel::SelectVLDSTLane(SDNode *N, bool IsLoad,
                                         bool isUpdating, unsigned NumVecs,
                                         const uint16_t *DOpcodes,
                                         const uint16_t *QOpcodes) {
  assert(NumVecs >=2 && NumVecs <= 4 && "VLDSTLane NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();

  SDValue MemAddr, Align;
  unsigned AddrOpIdx = isUpdating ? 1 : 2;
  unsigned Vec0Idx = 3; // AddrOpIdx + (isUpdating ? 2 : 1)
  if (!SelectAddrMode6(N, N->getOperand(AddrOpIdx), MemAddr, Align))
    return NULL;

  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();

  SDValue Chain = N->getOperand(0);
  unsigned Lane =
    cast<ConstantSDNode>(N->getOperand(Vec0Idx + NumVecs))->getZExtValue();
  EVT VT = N->getOperand(Vec0Idx).getValueType();
  bool is64BitVector = VT.is64BitVector();

  unsigned Alignment = 0;
  if (NumVecs != 3) {
    Alignment = cast<ConstantSDNode>(Align)->getZExtValue();
    unsigned NumBytes = NumVecs * VT.getVectorElementType().getSizeInBits()/8;
    if (Alignment > NumBytes)
      Alignment = NumBytes;
    if (Alignment < 8 && Alignment < NumBytes)
      Alignment = 0;
    // Alignment must be a power of two; make sure of that.
    Alignment = (Alignment & -Alignment);
    if (Alignment == 1)
      Alignment = 0;
  }
  Align = CurDAG->getTargetConstant(Alignment, MVT::i32);

  unsigned OpcodeIndex;
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("unhandled vld/vst lane type");
    // Double-register operations:
  case MVT::v8i8:  OpcodeIndex = 0; break;
  case MVT::v4i16: OpcodeIndex = 1; break;
  case MVT::v2f32:
  case MVT::v2i32: OpcodeIndex = 2; break;
    // Quad-register operations:
  case MVT::v8i16: OpcodeIndex = 0; break;
  case MVT::v4f32:
  case MVT::v4i32: OpcodeIndex = 1; break;
  }

  std::vector<EVT> ResTys;
  if (IsLoad) {
    unsigned ResTyElts = (NumVecs == 3) ? 4 : NumVecs;
    if (!is64BitVector)
      ResTyElts *= 2;
    ResTys.push_back(EVT::getVectorVT(*CurDAG->getContext(),
                                      MVT::i64, ResTyElts));
  }
  if (isUpdating)
    ResTys.push_back(MVT::i32);
  ResTys.push_back(MVT::Other);

  SDValue Pred = getAL(CurDAG);
  SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);

  SmallVector<SDValue, 8> Ops;
  Ops.push_back(MemAddr);
  Ops.push_back(Align);
  if (isUpdating) {
    SDValue Inc = N->getOperand(AddrOpIdx + 1);
    Ops.push_back(isa<ConstantSDNode>(Inc.getNode()) ? Reg0 : Inc);
  }

  SDValue SuperReg;
  SDValue V0 = N->getOperand(Vec0Idx + 0);
  SDValue V1 = N->getOperand(Vec0Idx + 1);
  if (NumVecs == 2) {
    if (is64BitVector)
      SuperReg = SDValue(PairDRegs(MVT::v2i64, V0, V1), 0);
    else
      SuperReg = SDValue(PairQRegs(MVT::v4i64, V0, V1), 0);
  } else {
    SDValue V2 = N->getOperand(Vec0Idx + 2);
    SDValue V3 = (NumVecs == 3)
      ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, dl, VT), 0)
      : N->getOperand(Vec0Idx + 3);
    if (is64BitVector)
      SuperReg = SDValue(QuadDRegs(MVT::v4i64, V0, V1, V2, V3), 0);
    else
      SuperReg = SDValue(QuadQRegs(MVT::v8i64, V0, V1, V2, V3), 0);
  }
  Ops.push_back(SuperReg);
  Ops.push_back(getI32Imm(Lane));
  Ops.push_back(Pred);
  Ops.push_back(Reg0);
  Ops.push_back(Chain);

  unsigned Opc = (is64BitVector ? DOpcodes[OpcodeIndex] :
                                  QOpcodes[OpcodeIndex]);
  SDNode *VLdLn = CurDAG->getMachineNode(Opc, dl, ResTys,
                                         Ops.data(), Ops.size());
  cast<MachineSDNode>(VLdLn)->setMemRefs(MemOp, MemOp + 1);
  if (!IsLoad)
    return VLdLn;

  // Extract the subregisters.
  SuperReg = SDValue(VLdLn, 0);
  assert(ARM::dsub_7 == ARM::dsub_0+7 &&
         ARM::qsub_3 == ARM::qsub_0+3 && "Unexpected subreg numbering");
  unsigned Sub0 = is64BitVector ? ARM::dsub_0 : ARM::qsub_0;
  for (unsigned Vec = 0; Vec < NumVecs; ++Vec)
    ReplaceUses(SDValue(N, Vec),
                CurDAG->getTargetExtractSubreg(Sub0 + Vec, dl, VT, SuperReg));
  ReplaceUses(SDValue(N, NumVecs), SDValue(VLdLn, 1));
  if (isUpdating)
    ReplaceUses(SDValue(N, NumVecs + 1), SDValue(VLdLn, 2));
  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectVLDDup(SDNode *N, bool isUpdating,
                                      unsigned NumVecs,
                                      const uint16_t *Opcodes) {
  assert(NumVecs >=2 && NumVecs <= 4 && "VLDDup NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();

  SDValue MemAddr, Align;
  if (!SelectAddrMode6(N, N->getOperand(1), MemAddr, Align))
    return NULL;

  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();

  SDValue Chain = N->getOperand(0);
  EVT VT = N->getValueType(0);

  unsigned Alignment = 0;
  if (NumVecs != 3) {
    Alignment = cast<ConstantSDNode>(Align)->getZExtValue();
    unsigned NumBytes = NumVecs * VT.getVectorElementType().getSizeInBits()/8;
    if (Alignment > NumBytes)
      Alignment = NumBytes;
    if (Alignment < 8 && Alignment < NumBytes)
      Alignment = 0;
    // Alignment must be a power of two; make sure of that.
    Alignment = (Alignment & -Alignment);
    if (Alignment == 1)
      Alignment = 0;
  }
  Align = CurDAG->getTargetConstant(Alignment, MVT::i32);

  unsigned OpcodeIndex;
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("unhandled vld-dup type");
  case MVT::v8i8:  OpcodeIndex = 0; break;
  case MVT::v4i16: OpcodeIndex = 1; break;
  case MVT::v2f32:
  case MVT::v2i32: OpcodeIndex = 2; break;
  }

  SDValue Pred = getAL(CurDAG);
  SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
  SDValue SuperReg;
  unsigned Opc = Opcodes[OpcodeIndex];
  SmallVector<SDValue, 6> Ops;
  Ops.push_back(MemAddr);
  Ops.push_back(Align);
  if (isUpdating) {
    // fixed-stride update instructions don't have an explicit writeback
    // operand. It's implicit in the opcode itself.
    SDValue Inc = N->getOperand(2);
    if (!isa<ConstantSDNode>(Inc.getNode()))
      Ops.push_back(Inc);
    // FIXME: VLD3 and VLD4 haven't been updated to that form yet.
    else if (NumVecs > 2)
      Ops.push_back(Reg0);
  }
  Ops.push_back(Pred);
  Ops.push_back(Reg0);
  Ops.push_back(Chain);

  unsigned ResTyElts = (NumVecs == 3) ? 4 : NumVecs;
  std::vector<EVT> ResTys;
  ResTys.push_back(EVT::getVectorVT(*CurDAG->getContext(), MVT::i64,ResTyElts));
  if (isUpdating)
    ResTys.push_back(MVT::i32);
  ResTys.push_back(MVT::Other);
  SDNode *VLdDup =
    CurDAG->getMachineNode(Opc, dl, ResTys, Ops.data(), Ops.size());
  cast<MachineSDNode>(VLdDup)->setMemRefs(MemOp, MemOp + 1);
  SuperReg = SDValue(VLdDup, 0);

  // Extract the subregisters.
  assert(ARM::dsub_7 == ARM::dsub_0+7 && "Unexpected subreg numbering");
  unsigned SubIdx = ARM::dsub_0;
  for (unsigned Vec = 0; Vec < NumVecs; ++Vec)
    ReplaceUses(SDValue(N, Vec),
                CurDAG->getTargetExtractSubreg(SubIdx+Vec, dl, VT, SuperReg));
  ReplaceUses(SDValue(N, NumVecs), SDValue(VLdDup, 1));
  if (isUpdating)
    ReplaceUses(SDValue(N, NumVecs + 1), SDValue(VLdDup, 2));
  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectVTBL(SDNode *N, bool IsExt, unsigned NumVecs,
                                    unsigned Opc) {
  assert(NumVecs >= 2 && NumVecs <= 4 && "VTBL NumVecs out-of-range");
  DebugLoc dl = N->getDebugLoc();
  EVT VT = N->getValueType(0);
  unsigned FirstTblReg = IsExt ? 2 : 1;

  // Form a REG_SEQUENCE to force register allocation.
  SDValue RegSeq;
  SDValue V0 = N->getOperand(FirstTblReg + 0);
  SDValue V1 = N->getOperand(FirstTblReg + 1);
  if (NumVecs == 2)
    RegSeq = SDValue(PairDRegs(MVT::v16i8, V0, V1), 0);
  else {
    SDValue V2 = N->getOperand(FirstTblReg + 2);
    // If it's a vtbl3, form a quad D-register and leave the last part as
    // an undef.
    SDValue V3 = (NumVecs == 3)
      ? SDValue(CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, dl, VT), 0)
      : N->getOperand(FirstTblReg + 3);
    RegSeq = SDValue(QuadDRegs(MVT::v4i64, V0, V1, V2, V3), 0);
  }

  SmallVector<SDValue, 6> Ops;
  if (IsExt)
    Ops.push_back(N->getOperand(1));
  Ops.push_back(RegSeq);
  Ops.push_back(N->getOperand(FirstTblReg + NumVecs));
  Ops.push_back(getAL(CurDAG)); // predicate
  Ops.push_back(CurDAG->getRegister(0, MVT::i32)); // predicate register
  return CurDAG->getMachineNode(Opc, dl, VT, Ops.data(), Ops.size());
}

SDNode *ARMDAGToDAGISel::SelectV6T2BitfieldExtractOp(SDNode *N,
                                                     bool isSigned) {
  if (!Subtarget->hasV6T2Ops())
    return NULL;

  unsigned Opc = isSigned ? (Subtarget->isThumb() ? ARM::t2SBFX : ARM::SBFX)
    : (Subtarget->isThumb() ? ARM::t2UBFX : ARM::UBFX);


  // For unsigned extracts, check for a shift right and mask
  unsigned And_imm = 0;
  if (N->getOpcode() == ISD::AND) {
    if (isOpcWithIntImmediate(N, ISD::AND, And_imm)) {

      // The immediate is a mask of the low bits iff imm & (imm+1) == 0
      if (And_imm & (And_imm + 1))
        return NULL;

      unsigned Srl_imm = 0;
      if (isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::SRL,
                                Srl_imm)) {
        assert(Srl_imm > 0 && Srl_imm < 32 && "bad amount in shift node!");

        // Note: The width operand is encoded as width-1.
        unsigned Width = CountTrailingOnes_32(And_imm) - 1;
        unsigned LSB = Srl_imm;
        SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
        SDValue Ops[] = { N->getOperand(0).getOperand(0),
                          CurDAG->getTargetConstant(LSB, MVT::i32),
                          CurDAG->getTargetConstant(Width, MVT::i32),
          getAL(CurDAG), Reg0 };
        return CurDAG->SelectNodeTo(N, Opc, MVT::i32, Ops, 5);
      }
    }
    return NULL;
  }

  // Otherwise, we're looking for a shift of a shift
  unsigned Shl_imm = 0;
  if (isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::SHL, Shl_imm)) {
    assert(Shl_imm > 0 && Shl_imm < 32 && "bad amount in shift node!");
    unsigned Srl_imm = 0;
    if (isInt32Immediate(N->getOperand(1), Srl_imm)) {
      assert(Srl_imm > 0 && Srl_imm < 32 && "bad amount in shift node!");
      // Note: The width operand is encoded as width-1.
      unsigned Width = 32 - Srl_imm - 1;
      int LSB = Srl_imm - Shl_imm;
      if (LSB < 0)
        return NULL;
      SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
      SDValue Ops[] = { N->getOperand(0).getOperand(0),
                        CurDAG->getTargetConstant(LSB, MVT::i32),
                        CurDAG->getTargetConstant(Width, MVT::i32),
                        getAL(CurDAG), Reg0 };
      return CurDAG->SelectNodeTo(N, Opc, MVT::i32, Ops, 5);
    }
  }
  return NULL;
}

SDNode *ARMDAGToDAGISel::
SelectT2CMOVShiftOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                    ARMCC::CondCodes CCVal, SDValue CCR, SDValue InFlag) {
  SDValue CPTmp0;
  SDValue CPTmp1;
  if (SelectT2ShifterOperandReg(TrueVal, CPTmp0, CPTmp1)) {
    unsigned SOVal = cast<ConstantSDNode>(CPTmp1)->getZExtValue();
    unsigned SOShOp = ARM_AM::getSORegShOp(SOVal);
    unsigned Opc = 0;
    switch (SOShOp) {
    case ARM_AM::lsl: Opc = ARM::t2MOVCClsl; break;
    case ARM_AM::lsr: Opc = ARM::t2MOVCClsr; break;
    case ARM_AM::asr: Opc = ARM::t2MOVCCasr; break;
    case ARM_AM::ror: Opc = ARM::t2MOVCCror; break;
    default:
      llvm_unreachable("Unknown so_reg opcode!");
    }
    SDValue SOShImm =
      CurDAG->getTargetConstant(ARM_AM::getSORegOffset(SOVal), MVT::i32);
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, CPTmp0, SOShImm, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N, Opc, MVT::i32,Ops, 6);
  }
  return 0;
}

SDNode *ARMDAGToDAGISel::
SelectARMCMOVShiftOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                     ARMCC::CondCodes CCVal, SDValue CCR, SDValue InFlag) {
  SDValue CPTmp0;
  SDValue CPTmp1;
  SDValue CPTmp2;
  if (SelectImmShifterOperand(TrueVal, CPTmp0, CPTmp2)) {
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, CPTmp0, CPTmp2, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N, ARM::MOVCCsi, MVT::i32, Ops, 6);
  }

  if (SelectRegShifterOperand(TrueVal, CPTmp0, CPTmp1, CPTmp2)) {
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, CPTmp0, CPTmp1, CPTmp2, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N, ARM::MOVCCsr, MVT::i32, Ops, 7);
  }
  return 0;
}

SDNode *ARMDAGToDAGISel::
SelectT2CMOVImmOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                  ARMCC::CondCodes CCVal, SDValue CCR, SDValue InFlag) {
  ConstantSDNode *T = dyn_cast<ConstantSDNode>(TrueVal);
  if (!T)
    return 0;

  unsigned Opc = 0;
  unsigned TrueImm = T->getZExtValue();
  if (is_t2_so_imm(TrueImm)) {
    Opc = ARM::t2MOVCCi;
  } else if (TrueImm <= 0xffff) {
    Opc = ARM::t2MOVCCi16;
  } else if (is_t2_so_imm_not(TrueImm)) {
    TrueImm = ~TrueImm;
    Opc = ARM::t2MVNCCi;
  } else if (TrueVal.getNode()->hasOneUse() && Subtarget->hasV6T2Ops()) {
    // Large immediate.
    Opc = ARM::t2MOVCCi32imm;
  }

  if (Opc) {
    SDValue True = CurDAG->getTargetConstant(TrueImm, MVT::i32);
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, True, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N, Opc, MVT::i32, Ops, 5);
  }

  return 0;
}

SDNode *ARMDAGToDAGISel::
SelectARMCMOVImmOp(SDNode *N, SDValue FalseVal, SDValue TrueVal,
                   ARMCC::CondCodes CCVal, SDValue CCR, SDValue InFlag) {
  ConstantSDNode *T = dyn_cast<ConstantSDNode>(TrueVal);
  if (!T)
    return 0;

  unsigned Opc = 0;
  unsigned TrueImm = T->getZExtValue();
  bool isSoImm = is_so_imm(TrueImm);
  if (isSoImm) {
    Opc = ARM::MOVCCi;
  } else if (Subtarget->hasV6T2Ops() && TrueImm <= 0xffff) {
    Opc = ARM::MOVCCi16;
  } else if (is_so_imm_not(TrueImm)) {
    TrueImm = ~TrueImm;
    Opc = ARM::MVNCCi;
  } else if (TrueVal.getNode()->hasOneUse() &&
             (Subtarget->hasV6T2Ops() || ARM_AM::isSOImmTwoPartVal(TrueImm))) {
    // Large immediate.
    Opc = ARM::MOVCCi32imm;
  }

  if (Opc) {
    SDValue True = CurDAG->getTargetConstant(TrueImm, MVT::i32);
    SDValue CC = CurDAG->getTargetConstant(CCVal, MVT::i32);
    SDValue Ops[] = { FalseVal, True, CC, CCR, InFlag };
    return CurDAG->SelectNodeTo(N, Opc, MVT::i32, Ops, 5);
  }

  return 0;
}

SDNode *ARMDAGToDAGISel::SelectCMOVOp(SDNode *N) {
  EVT VT = N->getValueType(0);
  SDValue FalseVal = N->getOperand(0);
  SDValue TrueVal  = N->getOperand(1);
  SDValue CC = N->getOperand(2);
  SDValue CCR = N->getOperand(3);
  SDValue InFlag = N->getOperand(4);
  assert(CC.getOpcode() == ISD::Constant);
  assert(CCR.getOpcode() == ISD::Register);
  ARMCC::CondCodes CCVal =
    (ARMCC::CondCodes)cast<ConstantSDNode>(CC)->getZExtValue();

  if (!Subtarget->isThumb1Only() && VT == MVT::i32) {
    // Pattern: (ARMcmov:i32 GPR:i32:$false, so_reg:i32:$true, (imm:i32):$cc)
    // Emits: (MOVCCs:i32 GPR:i32:$false, so_reg:i32:$true, (imm:i32):$cc)
    // Pattern complexity = 18  cost = 1  size = 0
    if (Subtarget->isThumb()) {
      SDNode *Res = SelectT2CMOVShiftOp(N, FalseVal, TrueVal,
                                        CCVal, CCR, InFlag);
      if (!Res)
        Res = SelectT2CMOVShiftOp(N, TrueVal, FalseVal,
                               ARMCC::getOppositeCondition(CCVal), CCR, InFlag);
      if (Res)
        return Res;
    } else {
      SDNode *Res = SelectARMCMOVShiftOp(N, FalseVal, TrueVal,
                                         CCVal, CCR, InFlag);
      if (!Res)
        Res = SelectARMCMOVShiftOp(N, TrueVal, FalseVal,
                               ARMCC::getOppositeCondition(CCVal), CCR, InFlag);
      if (Res)
        return Res;
    }

    // Pattern: (ARMcmov:i32 GPR:i32:$false,
    //             (imm:i32)<<P:Pred_so_imm>>:$true,
    //             (imm:i32):$cc)
    // Emits: (MOVCCi:i32 GPR:i32:$false,
    //           (so_imm:i32 (imm:i32):$true), (imm:i32):$cc)
    // Pattern complexity = 10  cost = 1  size = 0
    if (Subtarget->isThumb()) {
      SDNode *Res = SelectT2CMOVImmOp(N, FalseVal, TrueVal,
                                        CCVal, CCR, InFlag);
      if (!Res)
        Res = SelectT2CMOVImmOp(N, TrueVal, FalseVal,
                               ARMCC::getOppositeCondition(CCVal), CCR, InFlag);
      if (Res)
        return Res;
    } else {
      SDNode *Res = SelectARMCMOVImmOp(N, FalseVal, TrueVal,
                                         CCVal, CCR, InFlag);
      if (!Res)
        Res = SelectARMCMOVImmOp(N, TrueVal, FalseVal,
                               ARMCC::getOppositeCondition(CCVal), CCR, InFlag);
      if (Res)
        return Res;
    }
  }

  // Pattern: (ARMcmov:i32 GPR:i32:$false, GPR:i32:$true, (imm:i32):$cc)
  // Emits: (MOVCCr:i32 GPR:i32:$false, GPR:i32:$true, (imm:i32):$cc)
  // Pattern complexity = 6  cost = 1  size = 0
  //
  // Pattern: (ARMcmov:i32 GPR:i32:$false, GPR:i32:$true, (imm:i32):$cc)
  // Emits: (tMOVCCr:i32 GPR:i32:$false, GPR:i32:$true, (imm:i32):$cc)
  // Pattern complexity = 6  cost = 11  size = 0
  //
  // Also VMOVScc and VMOVDcc.
  SDValue Tmp2 = CurDAG->getTargetConstant(CCVal, MVT::i32);
  SDValue Ops[] = { FalseVal, TrueVal, Tmp2, CCR, InFlag };
  unsigned Opc = 0;
  switch (VT.getSimpleVT().SimpleTy) {
  default: llvm_unreachable("Illegal conditional move type!");
  case MVT::i32:
    Opc = Subtarget->isThumb()
      ? (Subtarget->hasThumb2() ? ARM::t2MOVCCr : ARM::tMOVCCr_pseudo)
      : ARM::MOVCCr;
    break;
  case MVT::f32:
    Opc = ARM::VMOVScc;
    break;
  case MVT::f64:
    Opc = ARM::VMOVDcc;
    break;
  }
  return CurDAG->SelectNodeTo(N, Opc, VT, Ops, 5);
}

/// Target-specific DAG combining for ISD::XOR.
/// Target-independent combining lowers SELECT_CC nodes of the form
/// select_cc setg[ge] X,  0,  X, -X
/// select_cc setgt    X, -1,  X, -X
/// select_cc setl[te] X,  0, -X,  X
/// select_cc setlt    X,  1, -X,  X
/// which represent Integer ABS into:
/// Y = sra (X, size(X)-1); xor (add (X, Y), Y)
/// ARM instruction selection detects the latter and matches it to
/// ARM::ABS or ARM::t2ABS machine node.
SDNode *ARMDAGToDAGISel::SelectABSOp(SDNode *N){
  SDValue XORSrc0 = N->getOperand(0);
  SDValue XORSrc1 = N->getOperand(1);
  EVT VT = N->getValueType(0);

  if (Subtarget->isThumb1Only())
    return NULL;

  if (XORSrc0.getOpcode() != ISD::ADD || XORSrc1.getOpcode() != ISD::SRA)
    return NULL;

  SDValue ADDSrc0 = XORSrc0.getOperand(0);
  SDValue ADDSrc1 = XORSrc0.getOperand(1);
  SDValue SRASrc0 = XORSrc1.getOperand(0);
  SDValue SRASrc1 = XORSrc1.getOperand(1);
  ConstantSDNode *SRAConstant =  dyn_cast<ConstantSDNode>(SRASrc1);
  EVT XType = SRASrc0.getValueType();
  unsigned Size = XType.getSizeInBits() - 1;

  if (ADDSrc1 == XORSrc1 && ADDSrc0 == SRASrc0 &&
      XType.isInteger() && SRAConstant != NULL &&
      Size == SRAConstant->getZExtValue()) {
    unsigned Opcode = Subtarget->isThumb2() ? ARM::t2ABS : ARM::ABS;
    return CurDAG->SelectNodeTo(N, Opcode, VT, ADDSrc0);
  }

  return NULL;
}

SDNode *ARMDAGToDAGISel::SelectConcatVector(SDNode *N) {
  // The only time a CONCAT_VECTORS operation can have legal types is when
  // two 64-bit vectors are concatenated to a 128-bit vector.
  EVT VT = N->getValueType(0);
  if (!VT.is128BitVector() || N->getNumOperands() != 2)
    llvm_unreachable("unexpected CONCAT_VECTORS");
  return PairDRegs(VT, N->getOperand(0), N->getOperand(1));
}

SDNode *ARMDAGToDAGISel::SelectAtomic64(SDNode *Node, unsigned Opc) {
  SmallVector<SDValue, 6> Ops;
  Ops.push_back(Node->getOperand(1)); // Ptr
  Ops.push_back(Node->getOperand(2)); // Low part of Val1
  Ops.push_back(Node->getOperand(3)); // High part of Val1
  if (Opc == ARM::ATOMCMPXCHG6432) {
    Ops.push_back(Node->getOperand(4)); // Low part of Val2
    Ops.push_back(Node->getOperand(5)); // High part of Val2
  }
  Ops.push_back(Node->getOperand(0)); // Chain
  MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
  MemOp[0] = cast<MemSDNode>(Node)->getMemOperand();
  SDNode *ResNode = CurDAG->getMachineNode(Opc, Node->getDebugLoc(),
                                           MVT::i32, MVT::i32, MVT::Other,
                                           Ops.data() ,Ops.size());
  cast<MachineSDNode>(ResNode)->setMemRefs(MemOp, MemOp + 1);
  return ResNode;
}

SDNode *ARMDAGToDAGISel::Select(SDNode *N) {
  DebugLoc dl = N->getDebugLoc();

  if (N->isMachineOpcode())
    return NULL;   // Already selected.

  switch (N->getOpcode()) {
  default: break;
  case ISD::XOR: {
    // Select special operations if XOR node forms integer ABS pattern
    SDNode *ResNode = SelectABSOp(N);
    if (ResNode)
      return ResNode;
    // Other cases are autogenerated.
    break;
  }
  case ISD::Constant: {
    unsigned Val = cast<ConstantSDNode>(N)->getZExtValue();
    bool UseCP = true;
    if (Subtarget->hasThumb2())
      // Thumb2-aware targets have the MOVT instruction, so all immediates can
      // be done with MOV + MOVT, at worst.
      UseCP = 0;
    else {
      if (Subtarget->isThumb()) {
        UseCP = (Val > 255 &&                          // MOV
                 ~Val > 255 &&                         // MOV + MVN
                 !ARM_AM::isThumbImmShiftedVal(Val));  // MOV + LSL
      } else
        UseCP = (ARM_AM::getSOImmVal(Val) == -1 &&     // MOV
                 ARM_AM::getSOImmVal(~Val) == -1 &&    // MVN
                 !ARM_AM::isSOImmTwoPartVal(Val));     // two instrs.
    }

    if (UseCP) {
      SDValue CPIdx =
        CurDAG->getTargetConstantPool(ConstantInt::get(
                                  Type::getInt32Ty(*CurDAG->getContext()), Val),
                                      TLI.getPointerTy());

      SDNode *ResNode;
      if (Subtarget->isThumb1Only()) {
        SDValue Pred = getAL(CurDAG);
        SDValue PredReg = CurDAG->getRegister(0, MVT::i32);
        SDValue Ops[] = { CPIdx, Pred, PredReg, CurDAG->getEntryNode() };
        ResNode = CurDAG->getMachineNode(ARM::tLDRpci, dl, MVT::i32, MVT::Other,
                                         Ops, 4);
      } else {
        SDValue Ops[] = {
          CPIdx,
          CurDAG->getTargetConstant(0, MVT::i32),
          getAL(CurDAG),
          CurDAG->getRegister(0, MVT::i32),
          CurDAG->getEntryNode()
        };
        ResNode=CurDAG->getMachineNode(ARM::LDRcp, dl, MVT::i32, MVT::Other,
                                       Ops, 5);
      }
      ReplaceUses(SDValue(N, 0), SDValue(ResNode, 0));
      return NULL;
    }

    // Other cases are autogenerated.
    break;
  }
  case ISD::FrameIndex: {
    // Selects to ADDri FI, 0 which in turn will become ADDri SP, imm.
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, TLI.getPointerTy());
    if (Subtarget->isThumb1Only()) {
      SDValue Ops[] = { TFI, CurDAG->getTargetConstant(0, MVT::i32),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->SelectNodeTo(N, ARM::tADDrSPi, MVT::i32, Ops, 4);
    } else {
      unsigned Opc = ((Subtarget->isThumb() && Subtarget->hasThumb2()) ?
                      ARM::t2ADDri : ARM::ADDri);
      SDValue Ops[] = { TFI, CurDAG->getTargetConstant(0, MVT::i32),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->SelectNodeTo(N, Opc, MVT::i32, Ops, 5);
    }
  }
  case ISD::SRL:
    if (SDNode *I = SelectV6T2BitfieldExtractOp(N, false))
      return I;
    break;
  case ISD::SRA:
    if (SDNode *I = SelectV6T2BitfieldExtractOp(N, true))
      return I;
    break;
  case ISD::MUL:
    if (Subtarget->isThumb1Only())
      break;
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1))) {
      unsigned RHSV = C->getZExtValue();
      if (!RHSV) break;
      if (isPowerOf2_32(RHSV-1)) {  // 2^n+1?
        unsigned ShImm = Log2_32(RHSV-1);
        if (ShImm >= 32)
          break;
        SDValue V = N->getOperand(0);
        ShImm = ARM_AM::getSORegOpc(ARM_AM::lsl, ShImm);
        SDValue ShImmOp = CurDAG->getTargetConstant(ShImm, MVT::i32);
        SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
        if (Subtarget->isThumb()) {
          SDValue Ops[] = { V, V, ShImmOp, getAL(CurDAG), Reg0, Reg0 };
          return CurDAG->SelectNodeTo(N, ARM::t2ADDrs, MVT::i32, Ops, 6);
        } else {
          SDValue Ops[] = { V, V, Reg0, ShImmOp, getAL(CurDAG), Reg0, Reg0 };
          return CurDAG->SelectNodeTo(N, ARM::ADDrsi, MVT::i32, Ops, 7);
        }
      }
      if (isPowerOf2_32(RHSV+1)) {  // 2^n-1?
        unsigned ShImm = Log2_32(RHSV+1);
        if (ShImm >= 32)
          break;
        SDValue V = N->getOperand(0);
        ShImm = ARM_AM::getSORegOpc(ARM_AM::lsl, ShImm);
        SDValue ShImmOp = CurDAG->getTargetConstant(ShImm, MVT::i32);
        SDValue Reg0 = CurDAG->getRegister(0, MVT::i32);
        if (Subtarget->isThumb()) {
          SDValue Ops[] = { V, V, ShImmOp, getAL(CurDAG), Reg0, Reg0 };
          return CurDAG->SelectNodeTo(N, ARM::t2RSBrs, MVT::i32, Ops, 6);
        } else {
          SDValue Ops[] = { V, V, Reg0, ShImmOp, getAL(CurDAG), Reg0, Reg0 };
          return CurDAG->SelectNodeTo(N, ARM::RSBrsi, MVT::i32, Ops, 7);
        }
      }
    }
    break;
  case ISD::AND: {
    // Check for unsigned bitfield extract
    if (SDNode *I = SelectV6T2BitfieldExtractOp(N, false))
      return I;

    // (and (or x, c2), c1) and top 16-bits of c1 and c2 match, lower 16-bits
    // of c1 are 0xffff, and lower 16-bit of c2 are 0. That is, the top 16-bits
    // are entirely contributed by c2 and lower 16-bits are entirely contributed
    // by x. That's equal to (or (and x, 0xffff), (and c1, 0xffff0000)).
    // Select it to: "movt x, ((c1 & 0xffff) >> 16)
    EVT VT = N->getValueType(0);
    if (VT != MVT::i32)
      break;
    unsigned Opc = (Subtarget->isThumb() && Subtarget->hasThumb2())
      ? ARM::t2MOVTi16
      : (Subtarget->hasV6T2Ops() ? ARM::MOVTi16 : 0);
    if (!Opc)
      break;
    SDValue N0 = N->getOperand(0), N1 = N->getOperand(1);
    ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
    if (!N1C)
      break;
    if (N0.getOpcode() == ISD::OR && N0.getNode()->hasOneUse()) {
      SDValue N2 = N0.getOperand(1);
      ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N2);
      if (!N2C)
        break;
      unsigned N1CVal = N1C->getZExtValue();
      unsigned N2CVal = N2C->getZExtValue();
      if ((N1CVal & 0xffff0000U) == (N2CVal & 0xffff0000U) &&
          (N1CVal & 0xffffU) == 0xffffU &&
          (N2CVal & 0xffffU) == 0x0U) {
        SDValue Imm16 = CurDAG->getTargetConstant((N2CVal & 0xFFFF0000U) >> 16,
                                                  MVT::i32);
        SDValue Ops[] = { N0.getOperand(0), Imm16,
                          getAL(CurDAG), CurDAG->getRegister(0, MVT::i32) };
        return CurDAG->getMachineNode(Opc, dl, VT, Ops, 4);
      }
    }
    break;
  }
  case ARMISD::VMOVRRD:
    return CurDAG->getMachineNode(ARM::VMOVRRD, dl, MVT::i32, MVT::i32,
                                  N->getOperand(0), getAL(CurDAG),
                                  CurDAG->getRegister(0, MVT::i32));
  case ISD::UMUL_LOHI: {
    if (Subtarget->isThumb1Only())
      break;
    if (Subtarget->isThumb()) {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(ARM::t2UMULL, dl, MVT::i32, MVT::i32,Ops,4);
    } else {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(Subtarget->hasV6Ops() ?
                                    ARM::UMULL : ARM::UMULLv5,
                                    dl, MVT::i32, MVT::i32, Ops, 5);
    }
  }
  case ISD::SMUL_LOHI: {
    if (Subtarget->isThumb1Only())
      break;
    if (Subtarget->isThumb()) {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(ARM::t2SMULL, dl, MVT::i32, MVT::i32,Ops,4);
    } else {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),
                        getAL(CurDAG), CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(Subtarget->hasV6Ops() ?
                                    ARM::SMULL : ARM::SMULLv5,
                                    dl, MVT::i32, MVT::i32, Ops, 5);
    }
  }
  case ARMISD::UMLAL:{
    if (Subtarget->isThumb()) {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1), N->getOperand(2),
                        N->getOperand(3), getAL(CurDAG),
                        CurDAG->getRegister(0, MVT::i32)};
      return CurDAG->getMachineNode(ARM::t2UMLAL, dl, MVT::i32, MVT::i32, Ops, 6);
    }else{
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1), N->getOperand(2),
                        N->getOperand(3), getAL(CurDAG),
                        CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(Subtarget->hasV6Ops() ?
                                      ARM::UMLAL : ARM::UMLALv5,
                                      dl, MVT::i32, MVT::i32, Ops, 7);
    }
  }
  case ARMISD::SMLAL:{
    if (Subtarget->isThumb()) {
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1), N->getOperand(2),
                        N->getOperand(3), getAL(CurDAG),
                        CurDAG->getRegister(0, MVT::i32)};
      return CurDAG->getMachineNode(ARM::t2SMLAL, dl, MVT::i32, MVT::i32, Ops, 6);
    }else{
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1), N->getOperand(2),
                        N->getOperand(3), getAL(CurDAG),
                        CurDAG->getRegister(0, MVT::i32),
                        CurDAG->getRegister(0, MVT::i32) };
      return CurDAG->getMachineNode(Subtarget->hasV6Ops() ?
                                      ARM::SMLAL : ARM::SMLALv5,
                                      dl, MVT::i32, MVT::i32, Ops, 7);
    }
  }
  case ISD::LOAD: {
    SDNode *ResNode = 0;
    if (Subtarget->isThumb() && Subtarget->hasThumb2())
      ResNode = SelectT2IndexedLoad(N);
    else
      ResNode = SelectARMIndexedLoad(N);
    if (ResNode)
      return ResNode;
    // Other cases are autogenerated.
    break;
  }
  case ARMISD::BRCOND: {
    // Pattern: (ARMbrcond:void (bb:Other):$dst, (imm:i32):$cc)
    // Emits: (Bcc:void (bb:Other):$dst, (imm:i32):$cc)
    // Pattern complexity = 6  cost = 1  size = 0

    // Pattern: (ARMbrcond:void (bb:Other):$dst, (imm:i32):$cc)
    // Emits: (tBcc:void (bb:Other):$dst, (imm:i32):$cc)
    // Pattern complexity = 6  cost = 1  size = 0

    // Pattern: (ARMbrcond:void (bb:Other):$dst, (imm:i32):$cc)
    // Emits: (t2Bcc:void (bb:Other):$dst, (imm:i32):$cc)
    // Pattern complexity = 6  cost = 1  size = 0

    unsigned Opc = Subtarget->isThumb() ?
      ((Subtarget->hasThumb2()) ? ARM::t2Bcc : ARM::tBcc) : ARM::Bcc;
    SDValue Chain = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue N2 = N->getOperand(2);
    SDValue N3 = N->getOperand(3);
    SDValue InFlag = N->getOperand(4);
    assert(N1.getOpcode() == ISD::BasicBlock);
    assert(N2.getOpcode() == ISD::Constant);
    assert(N3.getOpcode() == ISD::Register);

    SDValue Tmp2 = CurDAG->getTargetConstant(((unsigned)
                               cast<ConstantSDNode>(N2)->getZExtValue()),
                               MVT::i32);
    SDValue Ops[] = { N1, Tmp2, N3, Chain, InFlag };
    SDNode *ResNode = CurDAG->getMachineNode(Opc, dl, MVT::Other,
                                             MVT::Glue, Ops, 5);
    Chain = SDValue(ResNode, 0);
    if (N->getNumValues() == 2) {
      InFlag = SDValue(ResNode, 1);
      ReplaceUses(SDValue(N, 1), InFlag);
    }
    ReplaceUses(SDValue(N, 0),
                SDValue(Chain.getNode(), Chain.getResNo()));
    return NULL;
  }
  case ARMISD::CMOV:
    return SelectCMOVOp(N);
  case ARMISD::VZIP: {
    unsigned Opc = 0;
    EVT VT = N->getValueType(0);
    switch (VT.getSimpleVT().SimpleTy) {
    default: return NULL;
    case MVT::v8i8:  Opc = ARM::VZIPd8; break;
    case MVT::v4i16: Opc = ARM::VZIPd16; break;
    case MVT::v2f32:
    // vzip.32 Dd, Dm is a pseudo-instruction expanded to vtrn.32 Dd, Dm.
    case MVT::v2i32: Opc = ARM::VTRNd32; break;
    case MVT::v16i8: Opc = ARM::VZIPq8; break;
    case MVT::v8i16: Opc = ARM::VZIPq16; break;
    case MVT::v4f32:
    case MVT::v4i32: Opc = ARM::VZIPq32; break;
    }
    SDValue Pred = getAL(CurDAG);
    SDValue PredReg = CurDAG->getRegister(0, MVT::i32);
    SDValue Ops[] = { N->getOperand(0), N->getOperand(1), Pred, PredReg };
    return CurDAG->getMachineNode(Opc, dl, VT, VT, Ops, 4);
  }
  case ARMISD::VUZP: {
    unsigned Opc = 0;
    EVT VT = N->getValueType(0);
    switch (VT.getSimpleVT().SimpleTy) {
    default: return NULL;
    case MVT::v8i8:  Opc = ARM::VUZPd8; break;
    case MVT::v4i16: Opc = ARM::VUZPd16; break;
    case MVT::v2f32:
    // vuzp.32 Dd, Dm is a pseudo-instruction expanded to vtrn.32 Dd, Dm.
    case MVT::v2i32: Opc = ARM::VTRNd32; break;
    case MVT::v16i8: Opc = ARM::VUZPq8; break;
    case MVT::v8i16: Opc = ARM::VUZPq16; break;
    case MVT::v4f32:
    case MVT::v4i32: Opc = ARM::VUZPq32; break;
    }
    SDValue Pred = getAL(CurDAG);
    SDValue PredReg = CurDAG->getRegister(0, MVT::i32);
    SDValue Ops[] = { N->getOperand(0), N->getOperand(1), Pred, PredReg };
    return CurDAG->getMachineNode(Opc, dl, VT, VT, Ops, 4);
  }
  case ARMISD::VTRN: {
    unsigned Opc = 0;
    EVT VT = N->getValueType(0);
    switch (VT.getSimpleVT().SimpleTy) {
    default: return NULL;
    case MVT::v8i8:  Opc = ARM::VTRNd8; break;
    case MVT::v4i16: Opc = ARM::VTRNd16; break;
    case MVT::v2f32:
    case MVT::v2i32: Opc = ARM::VTRNd32; break;
    case MVT::v16i8: Opc = ARM::VTRNq8; break;
    case MVT::v8i16: Opc = ARM::VTRNq16; break;
    case MVT::v4f32:
    case MVT::v4i32: Opc = ARM::VTRNq32; break;
    }
    SDValue Pred = getAL(CurDAG);
    SDValue PredReg = CurDAG->getRegister(0, MVT::i32);
    SDValue Ops[] = { N->getOperand(0), N->getOperand(1), Pred, PredReg };
    return CurDAG->getMachineNode(Opc, dl, VT, VT, Ops, 4);
  }
  case ARMISD::BUILD_VECTOR: {
    EVT VecVT = N->getValueType(0);
    EVT EltVT = VecVT.getVectorElementType();
    unsigned NumElts = VecVT.getVectorNumElements();
    if (EltVT == MVT::f64) {
      assert(NumElts == 2 && "unexpected type for BUILD_VECTOR");
      return PairDRegs(VecVT, N->getOperand(0), N->getOperand(1));
    }
    assert(EltVT == MVT::f32 && "unexpected type for BUILD_VECTOR");
    if (NumElts == 2)
      return PairSRegs(VecVT, N->getOperand(0), N->getOperand(1));
    assert(NumElts == 4 && "unexpected type for BUILD_VECTOR");
    return QuadSRegs(VecVT, N->getOperand(0), N->getOperand(1),
                     N->getOperand(2), N->getOperand(3));
  }

  case ARMISD::VLD2DUP: {
    static const uint16_t Opcodes[] = { ARM::VLD2DUPd8, ARM::VLD2DUPd16,
                                        ARM::VLD2DUPd32 };
    return SelectVLDDup(N, false, 2, Opcodes);
  }

  case ARMISD::VLD3DUP: {
    static const uint16_t Opcodes[] = { ARM::VLD3DUPd8Pseudo,
                                        ARM::VLD3DUPd16Pseudo,
                                        ARM::VLD3DUPd32Pseudo };
    return SelectVLDDup(N, false, 3, Opcodes);
  }

  case ARMISD::VLD4DUP: {
    static const uint16_t Opcodes[] = { ARM::VLD4DUPd8Pseudo,
                                        ARM::VLD4DUPd16Pseudo,
                                        ARM::VLD4DUPd32Pseudo };
    return SelectVLDDup(N, false, 4, Opcodes);
  }

  case ARMISD::VLD2DUP_UPD: {
    static const uint16_t Opcodes[] = { ARM::VLD2DUPd8wb_fixed,
                                        ARM::VLD2DUPd16wb_fixed,
                                        ARM::VLD2DUPd32wb_fixed };
    return SelectVLDDup(N, true, 2, Opcodes);
  }

  case ARMISD::VLD3DUP_UPD: {
    static const uint16_t Opcodes[] = { ARM::VLD3DUPd8Pseudo_UPD,
                                        ARM::VLD3DUPd16Pseudo_UPD,
                                        ARM::VLD3DUPd32Pseudo_UPD };
    return SelectVLDDup(N, true, 3, Opcodes);
  }

  case ARMISD::VLD4DUP_UPD: {
    static const uint16_t Opcodes[] = { ARM::VLD4DUPd8Pseudo_UPD,
                                        ARM::VLD4DUPd16Pseudo_UPD,
                                        ARM::VLD4DUPd32Pseudo_UPD };
    return SelectVLDDup(N, true, 4, Opcodes);
  }

  case ARMISD::VLD1_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VLD1d8wb_fixed,
                                         ARM::VLD1d16wb_fixed,
                                         ARM::VLD1d32wb_fixed,
                                         ARM::VLD1d64wb_fixed };
    static const uint16_t QOpcodes[] = { ARM::VLD1q8wb_fixed,
                                         ARM::VLD1q16wb_fixed,
                                         ARM::VLD1q32wb_fixed,
                                         ARM::VLD1q64wb_fixed };
    return SelectVLD(N, true, 1, DOpcodes, QOpcodes, 0);
  }

  case ARMISD::VLD2_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VLD2d8wb_fixed,
                                         ARM::VLD2d16wb_fixed,
                                         ARM::VLD2d32wb_fixed,
                                         ARM::VLD1q64wb_fixed};
    static const uint16_t QOpcodes[] = { ARM::VLD2q8PseudoWB_fixed,
                                         ARM::VLD2q16PseudoWB_fixed,
                                         ARM::VLD2q32PseudoWB_fixed };
    return SelectVLD(N, true, 2, DOpcodes, QOpcodes, 0);
  }

  case ARMISD::VLD3_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VLD3d8Pseudo_UPD,
                                         ARM::VLD3d16Pseudo_UPD,
                                         ARM::VLD3d32Pseudo_UPD,
                                         ARM::VLD1q64wb_fixed};
    static const uint16_t QOpcodes0[] = { ARM::VLD3q8Pseudo_UPD,
                                          ARM::VLD3q16Pseudo_UPD,
                                          ARM::VLD3q32Pseudo_UPD };
    static const uint16_t QOpcodes1[] = { ARM::VLD3q8oddPseudo_UPD,
                                          ARM::VLD3q16oddPseudo_UPD,
                                          ARM::VLD3q32oddPseudo_UPD };
    return SelectVLD(N, true, 3, DOpcodes, QOpcodes0, QOpcodes1);
  }

  case ARMISD::VLD4_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VLD4d8Pseudo_UPD,
                                         ARM::VLD4d16Pseudo_UPD,
                                         ARM::VLD4d32Pseudo_UPD,
                                         ARM::VLD1q64wb_fixed};
    static const uint16_t QOpcodes0[] = { ARM::VLD4q8Pseudo_UPD,
                                          ARM::VLD4q16Pseudo_UPD,
                                          ARM::VLD4q32Pseudo_UPD };
    static const uint16_t QOpcodes1[] = { ARM::VLD4q8oddPseudo_UPD,
                                          ARM::VLD4q16oddPseudo_UPD,
                                          ARM::VLD4q32oddPseudo_UPD };
    return SelectVLD(N, true, 4, DOpcodes, QOpcodes0, QOpcodes1);
  }

  case ARMISD::VLD2LN_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VLD2LNd8Pseudo_UPD,
                                         ARM::VLD2LNd16Pseudo_UPD,
                                         ARM::VLD2LNd32Pseudo_UPD };
    static const uint16_t QOpcodes[] = { ARM::VLD2LNq16Pseudo_UPD,
                                         ARM::VLD2LNq32Pseudo_UPD };
    return SelectVLDSTLane(N, true, true, 2, DOpcodes, QOpcodes);
  }

  case ARMISD::VLD3LN_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VLD3LNd8Pseudo_UPD,
                                         ARM::VLD3LNd16Pseudo_UPD,
                                         ARM::VLD3LNd32Pseudo_UPD };
    static const uint16_t QOpcodes[] = { ARM::VLD3LNq16Pseudo_UPD,
                                         ARM::VLD3LNq32Pseudo_UPD };
    return SelectVLDSTLane(N, true, true, 3, DOpcodes, QOpcodes);
  }

  case ARMISD::VLD4LN_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VLD4LNd8Pseudo_UPD,
                                         ARM::VLD4LNd16Pseudo_UPD,
                                         ARM::VLD4LNd32Pseudo_UPD };
    static const uint16_t QOpcodes[] = { ARM::VLD4LNq16Pseudo_UPD,
                                         ARM::VLD4LNq32Pseudo_UPD };
    return SelectVLDSTLane(N, true, true, 4, DOpcodes, QOpcodes);
  }

  case ARMISD::VST1_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VST1d8wb_fixed,
                                         ARM::VST1d16wb_fixed,
                                         ARM::VST1d32wb_fixed,
                                         ARM::VST1d64wb_fixed };
    static const uint16_t QOpcodes[] = { ARM::VST1q8wb_fixed,
                                         ARM::VST1q16wb_fixed,
                                         ARM::VST1q32wb_fixed,
                                         ARM::VST1q64wb_fixed };
    return SelectVST(N, true, 1, DOpcodes, QOpcodes, 0);
  }

  case ARMISD::VST2_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VST2d8wb_fixed,
                                         ARM::VST2d16wb_fixed,
                                         ARM::VST2d32wb_fixed,
                                         ARM::VST1q64wb_fixed};
    static const uint16_t QOpcodes[] = { ARM::VST2q8PseudoWB_fixed,
                                         ARM::VST2q16PseudoWB_fixed,
                                         ARM::VST2q32PseudoWB_fixed };
    return SelectVST(N, true, 2, DOpcodes, QOpcodes, 0);
  }

  case ARMISD::VST3_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VST3d8Pseudo_UPD,
                                         ARM::VST3d16Pseudo_UPD,
                                         ARM::VST3d32Pseudo_UPD,
                                         ARM::VST1d64TPseudoWB_fixed};
    static const uint16_t QOpcodes0[] = { ARM::VST3q8Pseudo_UPD,
                                          ARM::VST3q16Pseudo_UPD,
                                          ARM::VST3q32Pseudo_UPD };
    static const uint16_t QOpcodes1[] = { ARM::VST3q8oddPseudo_UPD,
                                          ARM::VST3q16oddPseudo_UPD,
                                          ARM::VST3q32oddPseudo_UPD };
    return SelectVST(N, true, 3, DOpcodes, QOpcodes0, QOpcodes1);
  }

  case ARMISD::VST4_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VST4d8Pseudo_UPD,
                                         ARM::VST4d16Pseudo_UPD,
                                         ARM::VST4d32Pseudo_UPD,
                                         ARM::VST1d64QPseudoWB_fixed};
    static const uint16_t QOpcodes0[] = { ARM::VST4q8Pseudo_UPD,
                                          ARM::VST4q16Pseudo_UPD,
                                          ARM::VST4q32Pseudo_UPD };
    static const uint16_t QOpcodes1[] = { ARM::VST4q8oddPseudo_UPD,
                                          ARM::VST4q16oddPseudo_UPD,
                                          ARM::VST4q32oddPseudo_UPD };
    return SelectVST(N, true, 4, DOpcodes, QOpcodes0, QOpcodes1);
  }

  case ARMISD::VST2LN_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VST2LNd8Pseudo_UPD,
                                         ARM::VST2LNd16Pseudo_UPD,
                                         ARM::VST2LNd32Pseudo_UPD };
    static const uint16_t QOpcodes[] = { ARM::VST2LNq16Pseudo_UPD,
                                         ARM::VST2LNq32Pseudo_UPD };
    return SelectVLDSTLane(N, false, true, 2, DOpcodes, QOpcodes);
  }

  case ARMISD::VST3LN_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VST3LNd8Pseudo_UPD,
                                         ARM::VST3LNd16Pseudo_UPD,
                                         ARM::VST3LNd32Pseudo_UPD };
    static const uint16_t QOpcodes[] = { ARM::VST3LNq16Pseudo_UPD,
                                         ARM::VST3LNq32Pseudo_UPD };
    return SelectVLDSTLane(N, false, true, 3, DOpcodes, QOpcodes);
  }

  case ARMISD::VST4LN_UPD: {
    static const uint16_t DOpcodes[] = { ARM::VST4LNd8Pseudo_UPD,
                                         ARM::VST4LNd16Pseudo_UPD,
                                         ARM::VST4LNd32Pseudo_UPD };
    static const uint16_t QOpcodes[] = { ARM::VST4LNq16Pseudo_UPD,
                                         ARM::VST4LNq32Pseudo_UPD };
    return SelectVLDSTLane(N, false, true, 4, DOpcodes, QOpcodes);
  }

  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
    switch (IntNo) {
    default:
      break;

    case Intrinsic::arm_ldrexd: {
      SDValue MemAddr = N->getOperand(2);
      DebugLoc dl = N->getDebugLoc();
      SDValue Chain = N->getOperand(0);

      unsigned NewOpc = ARM::LDREXD;
      if (Subtarget->isThumb() && Subtarget->hasThumb2())
        NewOpc = ARM::t2LDREXD;

      // arm_ldrexd returns a i64 value in {i32, i32}
      std::vector<EVT> ResTys;
      ResTys.push_back(MVT::i32);
      ResTys.push_back(MVT::i32);
      ResTys.push_back(MVT::Other);

      // place arguments in the right order
      SmallVector<SDValue, 7> Ops;
      Ops.push_back(MemAddr);
      Ops.push_back(getAL(CurDAG));
      Ops.push_back(CurDAG->getRegister(0, MVT::i32));
      Ops.push_back(Chain);
      SDNode *Ld = CurDAG->getMachineNode(NewOpc, dl, ResTys, Ops.data(),
                                          Ops.size());
      // Transfer memoperands.
      MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
      MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();
      cast<MachineSDNode>(Ld)->setMemRefs(MemOp, MemOp + 1);

      // Until there's support for specifing explicit register constraints
      // like the use of even/odd register pair, hardcode ldrexd to always
      // use the pair [R0, R1] to hold the load result.
      Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, ARM::R0,
                                   SDValue(Ld, 0), SDValue(0,0));
      Chain = CurDAG->getCopyToReg(Chain, dl, ARM::R1,
                                   SDValue(Ld, 1), Chain.getValue(1));

      // Remap uses.
      SDValue Glue = Chain.getValue(1);
      if (!SDValue(N, 0).use_empty()) {
        SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                                ARM::R0, MVT::i32, Glue);
        Glue = Result.getValue(2);
        ReplaceUses(SDValue(N, 0), Result);
      }
      if (!SDValue(N, 1).use_empty()) {
        SDValue Result = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                                ARM::R1, MVT::i32, Glue);
        Glue = Result.getValue(2);
        ReplaceUses(SDValue(N, 1), Result);
      }

      ReplaceUses(SDValue(N, 2), SDValue(Ld, 2));
      return NULL;
    }

    case Intrinsic::arm_strexd: {
      DebugLoc dl = N->getDebugLoc();
      SDValue Chain = N->getOperand(0);
      SDValue Val0 = N->getOperand(2);
      SDValue Val1 = N->getOperand(3);
      SDValue MemAddr = N->getOperand(4);

      // Until there's support for specifing explicit register constraints
      // like the use of even/odd register pair, hardcode strexd to always
      // use the pair [R2, R3] to hold the i64 (i32, i32) value to be stored.
      Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, ARM::R2, Val0,
                                   SDValue(0, 0));
      Chain = CurDAG->getCopyToReg(Chain, dl, ARM::R3, Val1, Chain.getValue(1));

      SDValue Glue = Chain.getValue(1);
      Val0 = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                    ARM::R2, MVT::i32, Glue);
      Glue = Val0.getValue(1);
      Val1 = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), dl,
                                    ARM::R3, MVT::i32, Glue);

      // Store exclusive double return a i32 value which is the return status
      // of the issued store.
      std::vector<EVT> ResTys;
      ResTys.push_back(MVT::i32);
      ResTys.push_back(MVT::Other);

      // place arguments in the right order
      SmallVector<SDValue, 7> Ops;
      Ops.push_back(Val0);
      Ops.push_back(Val1);
      Ops.push_back(MemAddr);
      Ops.push_back(getAL(CurDAG));
      Ops.push_back(CurDAG->getRegister(0, MVT::i32));
      Ops.push_back(Chain);

      unsigned NewOpc = ARM::STREXD;
      if (Subtarget->isThumb() && Subtarget->hasThumb2())
        NewOpc = ARM::t2STREXD;

      SDNode *St = CurDAG->getMachineNode(NewOpc, dl, ResTys, Ops.data(),
                                          Ops.size());
      // Transfer memoperands.
      MachineSDNode::mmo_iterator MemOp = MF->allocateMemRefsArray(1);
      MemOp[0] = cast<MemIntrinsicSDNode>(N)->getMemOperand();
      cast<MachineSDNode>(St)->setMemRefs(MemOp, MemOp + 1);

      return St;
    }

    case Intrinsic::arm_neon_vld1: {
      static const uint16_t DOpcodes[] = { ARM::VLD1d8, ARM::VLD1d16,
                                           ARM::VLD1d32, ARM::VLD1d64 };
      static const uint16_t QOpcodes[] = { ARM::VLD1q8, ARM::VLD1q16,
                                           ARM::VLD1q32, ARM::VLD1q64};
      return SelectVLD(N, false, 1, DOpcodes, QOpcodes, 0);
    }

    case Intrinsic::arm_neon_vld2: {
      static const uint16_t DOpcodes[] = { ARM::VLD2d8, ARM::VLD2d16,
                                           ARM::VLD2d32, ARM::VLD1q64 };
      static const uint16_t QOpcodes[] = { ARM::VLD2q8Pseudo, ARM::VLD2q16Pseudo,
                                           ARM::VLD2q32Pseudo };
      return SelectVLD(N, false, 2, DOpcodes, QOpcodes, 0);
    }

    case Intrinsic::arm_neon_vld3: {
      static const uint16_t DOpcodes[] = { ARM::VLD3d8Pseudo,
                                           ARM::VLD3d16Pseudo,
                                           ARM::VLD3d32Pseudo,
                                           ARM::VLD1d64TPseudo };
      static const uint16_t QOpcodes0[] = { ARM::VLD3q8Pseudo_UPD,
                                            ARM::VLD3q16Pseudo_UPD,
                                            ARM::VLD3q32Pseudo_UPD };
      static const uint16_t QOpcodes1[] = { ARM::VLD3q8oddPseudo,
                                            ARM::VLD3q16oddPseudo,
                                            ARM::VLD3q32oddPseudo };
      return SelectVLD(N, false, 3, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vld4: {
      static const uint16_t DOpcodes[] = { ARM::VLD4d8Pseudo,
                                           ARM::VLD4d16Pseudo,
                                           ARM::VLD4d32Pseudo,
                                           ARM::VLD1d64QPseudo };
      static const uint16_t QOpcodes0[] = { ARM::VLD4q8Pseudo_UPD,
                                            ARM::VLD4q16Pseudo_UPD,
                                            ARM::VLD4q32Pseudo_UPD };
      static const uint16_t QOpcodes1[] = { ARM::VLD4q8oddPseudo,
                                            ARM::VLD4q16oddPseudo,
                                            ARM::VLD4q32oddPseudo };
      return SelectVLD(N, false, 4, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vld2lane: {
      static const uint16_t DOpcodes[] = { ARM::VLD2LNd8Pseudo,
                                           ARM::VLD2LNd16Pseudo,
                                           ARM::VLD2LNd32Pseudo };
      static const uint16_t QOpcodes[] = { ARM::VLD2LNq16Pseudo,
                                           ARM::VLD2LNq32Pseudo };
      return SelectVLDSTLane(N, true, false, 2, DOpcodes, QOpcodes);
    }

    case Intrinsic::arm_neon_vld3lane: {
      static const uint16_t DOpcodes[] = { ARM::VLD3LNd8Pseudo,
                                           ARM::VLD3LNd16Pseudo,
                                           ARM::VLD3LNd32Pseudo };
      static const uint16_t QOpcodes[] = { ARM::VLD3LNq16Pseudo,
                                           ARM::VLD3LNq32Pseudo };
      return SelectVLDSTLane(N, true, false, 3, DOpcodes, QOpcodes);
    }

    case Intrinsic::arm_neon_vld4lane: {
      static const uint16_t DOpcodes[] = { ARM::VLD4LNd8Pseudo,
                                           ARM::VLD4LNd16Pseudo,
                                           ARM::VLD4LNd32Pseudo };
      static const uint16_t QOpcodes[] = { ARM::VLD4LNq16Pseudo,
                                           ARM::VLD4LNq32Pseudo };
      return SelectVLDSTLane(N, true, false, 4, DOpcodes, QOpcodes);
    }

    case Intrinsic::arm_neon_vst1: {
      static const uint16_t DOpcodes[] = { ARM::VST1d8, ARM::VST1d16,
                                           ARM::VST1d32, ARM::VST1d64 };
      static const uint16_t QOpcodes[] = { ARM::VST1q8, ARM::VST1q16,
                                           ARM::VST1q32, ARM::VST1q64 };
      return SelectVST(N, false, 1, DOpcodes, QOpcodes, 0);
    }

    case Intrinsic::arm_neon_vst2: {
      static const uint16_t DOpcodes[] = { ARM::VST2d8, ARM::VST2d16,
                                           ARM::VST2d32, ARM::VST1q64 };
      static uint16_t QOpcodes[] = { ARM::VST2q8Pseudo, ARM::VST2q16Pseudo,
                                     ARM::VST2q32Pseudo };
      return SelectVST(N, false, 2, DOpcodes, QOpcodes, 0);
    }

    case Intrinsic::arm_neon_vst3: {
      static const uint16_t DOpcodes[] = { ARM::VST3d8Pseudo,
                                           ARM::VST3d16Pseudo,
                                           ARM::VST3d32Pseudo,
                                           ARM::VST1d64TPseudo };
      static const uint16_t QOpcodes0[] = { ARM::VST3q8Pseudo_UPD,
                                            ARM::VST3q16Pseudo_UPD,
                                            ARM::VST3q32Pseudo_UPD };
      static const uint16_t QOpcodes1[] = { ARM::VST3q8oddPseudo,
                                            ARM::VST3q16oddPseudo,
                                            ARM::VST3q32oddPseudo };
      return SelectVST(N, false, 3, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vst4: {
      static const uint16_t DOpcodes[] = { ARM::VST4d8Pseudo,
                                           ARM::VST4d16Pseudo,
                                           ARM::VST4d32Pseudo,
                                           ARM::VST1d64QPseudo };
      static const uint16_t QOpcodes0[] = { ARM::VST4q8Pseudo_UPD,
                                            ARM::VST4q16Pseudo_UPD,
                                            ARM::VST4q32Pseudo_UPD };
      static const uint16_t QOpcodes1[] = { ARM::VST4q8oddPseudo,
                                            ARM::VST4q16oddPseudo,
                                            ARM::VST4q32oddPseudo };
      return SelectVST(N, false, 4, DOpcodes, QOpcodes0, QOpcodes1);
    }

    case Intrinsic::arm_neon_vst2lane: {
      static const uint16_t DOpcodes[] = { ARM::VST2LNd8Pseudo,
                                           ARM::VST2LNd16Pseudo,
                                           ARM::VST2LNd32Pseudo };
      static const uint16_t QOpcodes[] = { ARM::VST2LNq16Pseudo,
                                           ARM::VST2LNq32Pseudo };
      return SelectVLDSTLane(N, false, false, 2, DOpcodes, QOpcodes);
    }

    case Intrinsic::arm_neon_vst3lane: {
      static const uint16_t DOpcodes[] = { ARM::VST3LNd8Pseudo,
                                           ARM::VST3LNd16Pseudo,
                                           ARM::VST3LNd32Pseudo };
      static const uint16_t QOpcodes[] = { ARM::VST3LNq16Pseudo,
                                           ARM::VST3LNq32Pseudo };
      return SelectVLDSTLane(N, false, false, 3, DOpcodes, QOpcodes);
    }

    case Intrinsic::arm_neon_vst4lane: {
      static const uint16_t DOpcodes[] = { ARM::VST4LNd8Pseudo,
                                           ARM::VST4LNd16Pseudo,
                                           ARM::VST4LNd32Pseudo };
      static const uint16_t QOpcodes[] = { ARM::VST4LNq16Pseudo,
                                           ARM::VST4LNq32Pseudo };
      return SelectVLDSTLane(N, false, false, 4, DOpcodes, QOpcodes);
    }
    }
    break;
  }

  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IntNo = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    switch (IntNo) {
    default:
      break;

    case Intrinsic::arm_neon_vtbl2:
      return SelectVTBL(N, false, 2, ARM::VTBL2);
    case Intrinsic::arm_neon_vtbl3:
      return SelectVTBL(N, false, 3, ARM::VTBL3Pseudo);
    case Intrinsic::arm_neon_vtbl4:
      return SelectVTBL(N, false, 4, ARM::VTBL4Pseudo);

    case Intrinsic::arm_neon_vtbx2:
      return SelectVTBL(N, true, 2, ARM::VTBX2);
    case Intrinsic::arm_neon_vtbx3:
      return SelectVTBL(N, true, 3, ARM::VTBX3Pseudo);
    case Intrinsic::arm_neon_vtbx4:
      return SelectVTBL(N, true, 4, ARM::VTBX4Pseudo);
    }
    break;
  }

  case ARMISD::VTBL1: {
    DebugLoc dl = N->getDebugLoc();
    EVT VT = N->getValueType(0);
    SmallVector<SDValue, 6> Ops;

    Ops.push_back(N->getOperand(0));
    Ops.push_back(N->getOperand(1));
    Ops.push_back(getAL(CurDAG));                    // Predicate
    Ops.push_back(CurDAG->getRegister(0, MVT::i32)); // Predicate Register
    return CurDAG->getMachineNode(ARM::VTBL1, dl, VT, Ops.data(), Ops.size());
  }
  case ARMISD::VTBL2: {
    DebugLoc dl = N->getDebugLoc();
    EVT VT = N->getValueType(0);

    // Form a REG_SEQUENCE to force register allocation.
    SDValue V0 = N->getOperand(0);
    SDValue V1 = N->getOperand(1);
    SDValue RegSeq = SDValue(PairDRegs(MVT::v16i8, V0, V1), 0);

    SmallVector<SDValue, 6> Ops;
    Ops.push_back(RegSeq);
    Ops.push_back(N->getOperand(2));
    Ops.push_back(getAL(CurDAG));                    // Predicate
    Ops.push_back(CurDAG->getRegister(0, MVT::i32)); // Predicate Register
    return CurDAG->getMachineNode(ARM::VTBL2, dl, VT,
                                  Ops.data(), Ops.size());
  }

  case ISD::CONCAT_VECTORS:
    return SelectConcatVector(N);

  case ARMISD::ATOMOR64_DAG:
    return SelectAtomic64(N, ARM::ATOMOR6432);
  case ARMISD::ATOMXOR64_DAG:
    return SelectAtomic64(N, ARM::ATOMXOR6432);
  case ARMISD::ATOMADD64_DAG:
    return SelectAtomic64(N, ARM::ATOMADD6432);
  case ARMISD::ATOMSUB64_DAG:
    return SelectAtomic64(N, ARM::ATOMSUB6432);
  case ARMISD::ATOMNAND64_DAG:
    return SelectAtomic64(N, ARM::ATOMNAND6432);
  case ARMISD::ATOMAND64_DAG:
    return SelectAtomic64(N, ARM::ATOMAND6432);
  case ARMISD::ATOMSWAP64_DAG:
    return SelectAtomic64(N, ARM::ATOMSWAP6432);
  case ARMISD::ATOMCMPXCHG64_DAG:
    return SelectAtomic64(N, ARM::ATOMCMPXCHG6432);
  }

  return SelectCode(N);
}

bool ARMDAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op, char ConstraintCode,
                             std::vector<SDValue> &OutOps) {
  assert(ConstraintCode == 'm' && "unexpected asm memory constraint");
  // Require the address to be in a register.  That is safe for all ARM
  // variants and it is hard to do anything much smarter without knowing
  // how the operand is used.
  OutOps.push_back(Op);
  return false;
}

/// createARMISelDag - This pass converts a legalized DAG into a
/// ARM-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createARMISelDag(ARMBaseTargetMachine &TM,
                                     CodeGenOpt::Level OptLevel) {
  return new ARMDAGToDAGISel(TM, OptLevel);
}
