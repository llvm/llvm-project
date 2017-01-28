//===-- MipsSEISelDAGToDAG.cpp - A Dag to Dag Inst Selector for MipsSE ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Subclass of MipsDAGToDAGISel specialized for mips32/64.
//
//===----------------------------------------------------------------------===//

#include "MipsSEISelDAGToDAG.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "Mips.h"
#include "MipsAnalyzeImmediate.h"
#include "MipsMachineFunction.h"
#include "MipsRegisterInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "mips-isel"

bool MipsSEDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &static_cast<const MipsSubtarget &>(MF.getSubtarget());
  if (Subtarget->inMips16Mode())
    return false;
  return MipsDAGToDAGISel::runOnMachineFunction(MF);
}

void MipsSEDAGToDAGISel::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  SelectionDAGISel::getAnalysisUsage(AU);
}

void MipsSEDAGToDAGISel::addDSPCtrlRegOperands(bool IsDef, MachineInstr &MI,
                                               MachineFunction &MF) {
  MachineInstrBuilder MIB(MF, &MI);
  unsigned Mask = MI.getOperand(1).getImm();
  unsigned Flag =
      IsDef ? RegState::ImplicitDefine : RegState::Implicit | RegState::Undef;

  if (Mask & 1)
    MIB.addReg(Mips::DSPPos, Flag);

  if (Mask & 2)
    MIB.addReg(Mips::DSPSCount, Flag);

  if (Mask & 4)
    MIB.addReg(Mips::DSPCarry, Flag);

  if (Mask & 8)
    MIB.addReg(Mips::DSPOutFlag, Flag);

  if (Mask & 16)
    MIB.addReg(Mips::DSPCCond, Flag);

  if (Mask & 32)
    MIB.addReg(Mips::DSPEFI, Flag);
}

unsigned MipsSEDAGToDAGISel::getMSACtrlReg(const SDValue RegIdx) const {
  switch (cast<ConstantSDNode>(RegIdx)->getZExtValue()) {
  default:
    llvm_unreachable("Could not map int to register");
  case 0: return Mips::MSAIR;
  case 1: return Mips::MSACSR;
  case 2: return Mips::MSAAccess;
  case 3: return Mips::MSASave;
  case 4: return Mips::MSAModify;
  case 5: return Mips::MSARequest;
  case 6: return Mips::MSAMap;
  case 7: return Mips::MSAUnmap;
  }
}

bool MipsSEDAGToDAGISel::replaceUsesWithZeroReg(MachineRegisterInfo *MRI,
                                                const MachineInstr& MI) {
  unsigned DstReg = 0, ZeroReg = 0;

  // Check if MI is "addiu $dst, $zero, 0" or "daddiu $dst, $zero, 0".
  if ((MI.getOpcode() == Mips::ADDiu) &&
      (MI.getOperand(1).getReg() == Mips::ZERO) &&
      (MI.getOperand(2).isImm()) &&
      (MI.getOperand(2).getImm() == 0)) {
    DstReg = MI.getOperand(0).getReg();
    ZeroReg = Mips::ZERO;
  } else if ((MI.getOpcode() == Mips::DADDiu) &&
             (MI.getOperand(1).getReg() == Mips::ZERO_64) &&
             (MI.getOperand(2).isImm()) &&
             (MI.getOperand(2).getImm() == 0)) {
    DstReg = MI.getOperand(0).getReg();
    ZeroReg = Mips::ZERO_64;
  }

  if (!DstReg)
    return false;

  // Replace uses with ZeroReg.
  for (MachineRegisterInfo::use_iterator U = MRI->use_begin(DstReg),
       E = MRI->use_end(); U != E;) {
    MachineOperand &MO = *U;
    unsigned OpNo = U.getOperandNo();
    MachineInstr *MI = MO.getParent();
    ++U;

    // Do not replace if it is a phi's operand or is tied to def operand.
    if (MI->isPHI() || MI->isRegTiedToDefOperand(OpNo) || MI->isPseudo())
      continue;

    // Also, we have to check that the register class of the operand
    // contains the zero register.
    if (!MRI->getRegClass(MO.getReg())->contains(ZeroReg))
      continue;

    MO.setReg(ZeroReg);
  }

  return true;
}

void MipsSEDAGToDAGISel::initGlobalBaseReg(MachineFunction &MF) {
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  if (!MipsFI->globalBaseRegSet())
    return;

  MachineBasicBlock &MBB = MF.front();
  MachineBasicBlock::iterator I = MBB.begin();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL;
  unsigned V0, V1, GlobalBaseReg = MipsFI->getGlobalBaseReg();
  const TargetRegisterClass *RC;
  const MipsABIInfo &ABI = static_cast<const MipsTargetMachine &>(TM).getABI();
  RC = (ABI.IsN64()) ? &Mips::GPR64RegClass : &Mips::GPR32RegClass;

  V0 = RegInfo.createVirtualRegister(RC);
  V1 = RegInfo.createVirtualRegister(RC);

  if (ABI.IsN64()) {
    MF.getRegInfo().addLiveIn(Mips::T9_64);
    MBB.addLiveIn(Mips::T9_64);

    // lui $v0, %hi(%neg(%gp_rel(fname)))
    // daddu $v1, $v0, $t9
    // daddiu $globalbasereg, $v1, %lo(%neg(%gp_rel(fname)))
    const GlobalValue *FName = MF.getFunction();
    BuildMI(MBB, I, DL, TII.get(Mips::LUi64), V0)
      .addGlobalAddress(FName, 0, MipsII::MO_GPOFF_HI);
    BuildMI(MBB, I, DL, TII.get(Mips::DADDu), V1).addReg(V0)
      .addReg(Mips::T9_64);
    BuildMI(MBB, I, DL, TII.get(Mips::DADDiu), GlobalBaseReg).addReg(V1)
      .addGlobalAddress(FName, 0, MipsII::MO_GPOFF_LO);
    return;
  }

  if (!MF.getTarget().isPositionIndependent()) {
    // Set global register to __gnu_local_gp.
    //
    // lui   $v0, %hi(__gnu_local_gp)
    // addiu $globalbasereg, $v0, %lo(__gnu_local_gp)
    BuildMI(MBB, I, DL, TII.get(Mips::LUi), V0)
      .addExternalSymbol("__gnu_local_gp", MipsII::MO_ABS_HI);
    BuildMI(MBB, I, DL, TII.get(Mips::ADDiu), GlobalBaseReg).addReg(V0)
      .addExternalSymbol("__gnu_local_gp", MipsII::MO_ABS_LO);
    return;
  }

  MF.getRegInfo().addLiveIn(Mips::T9);
  MBB.addLiveIn(Mips::T9);

  if (ABI.IsN32()) {
    // lui $v0, %hi(%neg(%gp_rel(fname)))
    // addu $v1, $v0, $t9
    // addiu $globalbasereg, $v1, %lo(%neg(%gp_rel(fname)))
    const GlobalValue *FName = MF.getFunction();
    BuildMI(MBB, I, DL, TII.get(Mips::LUi), V0)
      .addGlobalAddress(FName, 0, MipsII::MO_GPOFF_HI);
    BuildMI(MBB, I, DL, TII.get(Mips::ADDu), V1).addReg(V0).addReg(Mips::T9);
    BuildMI(MBB, I, DL, TII.get(Mips::ADDiu), GlobalBaseReg).addReg(V1)
      .addGlobalAddress(FName, 0, MipsII::MO_GPOFF_LO);
    return;
  }

  assert(ABI.IsO32());

  // For O32 ABI, the following instruction sequence is emitted to initialize
  // the global base register:
  //
  //  0. lui   $2, %hi(_gp_disp)
  //  1. addiu $2, $2, %lo(_gp_disp)
  //  2. addu  $globalbasereg, $2, $t9
  //
  // We emit only the last instruction here.
  //
  // GNU linker requires that the first two instructions appear at the beginning
  // of a function and no instructions be inserted before or between them.
  // The two instructions are emitted during lowering to MC layer in order to
  // avoid any reordering.
  //
  // Register $2 (Mips::V0) is added to the list of live-in registers to ensure
  // the value instruction 1 (addiu) defines is valid when instruction 2 (addu)
  // reads it.
  MF.getRegInfo().addLiveIn(Mips::V0);
  MBB.addLiveIn(Mips::V0);
  BuildMI(MBB, I, DL, TII.get(Mips::ADDu), GlobalBaseReg)
    .addReg(Mips::V0).addReg(Mips::T9);
}

void MipsSEDAGToDAGISel::processFunctionAfterISel(MachineFunction &MF) {
  initGlobalBaseReg(MF);

  MachineRegisterInfo *MRI = &MF.getRegInfo();

  for (auto &MBB: MF) {
    for (auto &MI: MBB) {
      switch (MI.getOpcode()) {
      case Mips::RDDSP:
        addDSPCtrlRegOperands(false, MI, MF);
        break;
      case Mips::WRDSP:
        addDSPCtrlRegOperands(true, MI, MF);
        break;
      default:
        replaceUsesWithZeroReg(MRI, MI);
      }
    }
  }
}

void MipsSEDAGToDAGISel::selectAddESubE(unsigned MOp, SDValue InFlag,
                                        SDValue CmpLHS, const SDLoc &DL,
                                        SDNode *Node) const {
  unsigned Opc = InFlag.getOpcode(); (void)Opc;

  assert(((Opc == ISD::ADDC || Opc == ISD::ADDE) ||
          (Opc == ISD::SUBC || Opc == ISD::SUBE)) &&
         "(ADD|SUB)E flag operand must come from (ADD|SUB)C/E insn");

  unsigned SLTuOp = Mips::SLTu, ADDuOp = Mips::ADDu;
  if (Subtarget->isGP64bit()) {
    SLTuOp = Mips::SLTu64;
    ADDuOp = Mips::DADDu;
  }

  SDValue Ops[] = { CmpLHS, InFlag.getOperand(1) };
  SDValue LHS = Node->getOperand(0), RHS = Node->getOperand(1);
  EVT VT = LHS.getValueType();

  SDNode *Carry = CurDAG->getMachineNode(SLTuOp, DL, VT, Ops);

  if (Subtarget->isGP64bit()) {
    // On 64-bit targets, sltu produces an i64 but our backend currently says
    // that SLTu64 produces an i32. We need to fix this in the long run but for
    // now, just make the DAG type-correct by asserting the upper bits are zero.
    Carry = CurDAG->getMachineNode(Mips::SUBREG_TO_REG, DL, VT,
                                   CurDAG->getTargetConstant(0, DL, VT),
                                   SDValue(Carry, 0),
                                   CurDAG->getTargetConstant(Mips::sub_32, DL,
                                                             VT));
  }

  // Generate a second addition only if we know that RHS is not a
  // constant-zero node.
  SDNode *AddCarry = Carry;
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(RHS);
  if (!C || C->getZExtValue())
    AddCarry = CurDAG->getMachineNode(ADDuOp, DL, VT, SDValue(Carry, 0), RHS);

  CurDAG->SelectNodeTo(Node, MOp, VT, MVT::Glue, LHS, SDValue(AddCarry, 0));
}

/// Match frameindex
bool MipsSEDAGToDAGISel::selectAddrFrameIndex(SDValue Addr, SDValue &Base,
                                              SDValue &Offset) const {
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    EVT ValTy = Addr.getValueType();

    Base   = CurDAG->getTargetFrameIndex(FIN->getIndex(), ValTy);
    Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), ValTy);
    return true;
  }
  return false;
}

/// Match frameindex+offset and frameindex|offset
bool MipsSEDAGToDAGISel::selectAddrFrameIndexOffset(
    SDValue Addr, SDValue &Base, SDValue &Offset, unsigned OffsetBits,
    unsigned ShiftAmount = 0) const {
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1));
    if (isIntN(OffsetBits + ShiftAmount, CN->getSExtValue())) {
      EVT ValTy = Addr.getValueType();

      // If the first operand is a FI, get the TargetFI Node
      if (FrameIndexSDNode *FIN =
              dyn_cast<FrameIndexSDNode>(Addr.getOperand(0)))
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), ValTy);
      else {
        Base = Addr.getOperand(0);
        // If base is a FI, additional offset calculation is done in
        // eliminateFrameIndex, otherwise we need to check the alignment
        if (OffsetToAlignment(CN->getZExtValue(), 1ull << ShiftAmount) != 0)
          return false;
      }

      Offset = CurDAG->getTargetConstant(CN->getZExtValue(), SDLoc(Addr),
                                         ValTy);
      return true;
    }
  }
  return false;
}

/// ComplexPattern used on MipsInstrInfo
/// Used on Mips Load/Store instructions
bool MipsSEDAGToDAGISel::selectAddrRegImm(SDValue Addr, SDValue &Base,
                                          SDValue &Offset) const {
  // if Address is FI, get the TargetFrameIndex.
  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  // on PIC code Load GA
  if (Addr.getOpcode() == MipsISD::Wrapper) {
    Base   = Addr.getOperand(0);
    Offset = Addr.getOperand(1);
    return true;
  }

  if (!TM.isPositionIndependent()) {
    if ((Addr.getOpcode() == ISD::TargetExternalSymbol ||
        Addr.getOpcode() == ISD::TargetGlobalAddress))
      return false;
  }

  // Addresses of the form FI+const or FI|const
  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 16))
    return true;

  // Operand is a result from an ADD.
  if (Addr.getOpcode() == ISD::ADD) {
    // When loading from constant pools, load the lower address part in
    // the instruction itself. Example, instead of:
    //  lui $2, %hi($CPI1_0)
    //  addiu $2, $2, %lo($CPI1_0)
    //  lwc1 $f0, 0($2)
    // Generate:
    //  lui $2, %hi($CPI1_0)
    //  lwc1 $f0, %lo($CPI1_0)($2)
    if (Addr.getOperand(1).getOpcode() == MipsISD::Lo ||
        Addr.getOperand(1).getOpcode() == MipsISD::GPRel) {
      SDValue Opnd0 = Addr.getOperand(1).getOperand(0);
      if (isa<ConstantPoolSDNode>(Opnd0) || isa<GlobalAddressSDNode>(Opnd0) ||
          isa<JumpTableSDNode>(Opnd0)) {
        Base = Addr.getOperand(0);
        Offset = Opnd0;
        return true;
      }
    }
  }

  return false;
}

/// ComplexPattern used on MipsInstrInfo
/// Used on Mips Load/Store instructions
bool MipsSEDAGToDAGISel::selectAddrDefault(SDValue Addr, SDValue &Base,
                                           SDValue &Offset) const {
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), Addr.getValueType());
  return true;
}

bool MipsSEDAGToDAGISel::selectIntAddr(SDValue Addr, SDValue &Base,
                                       SDValue &Offset) const {
  return selectAddrRegImm(Addr, Base, Offset) ||
    selectAddrDefault(Addr, Base, Offset);
}

bool MipsSEDAGToDAGISel::selectAddrRegImm9(SDValue Addr, SDValue &Base,
                                           SDValue &Offset) const {
  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 9))
    return true;

  return false;
}

/// Used on microMIPS LWC2, LDC2, SWC2 and SDC2 instructions (11-bit offset)
bool MipsSEDAGToDAGISel::selectAddrRegImm11(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) const {
  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 11))
    return true;

  return false;
}

/// Used on microMIPS Load/Store unaligned instructions (12-bit offset)
bool MipsSEDAGToDAGISel::selectAddrRegImm12(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) const {
  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 12))
    return true;

  return false;
}

bool MipsSEDAGToDAGISel::selectAddrRegImm16(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) const {
  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 16))
    return true;

  return false;
}

bool MipsSEDAGToDAGISel::selectIntAddr11MM(SDValue Addr, SDValue &Base,
                                         SDValue &Offset) const {
  return selectAddrRegImm11(Addr, Base, Offset) ||
    selectAddrDefault(Addr, Base, Offset);
}

bool MipsSEDAGToDAGISel::selectIntAddr12MM(SDValue Addr, SDValue &Base,
                                         SDValue &Offset) const {
  return selectAddrRegImm12(Addr, Base, Offset) ||
    selectAddrDefault(Addr, Base, Offset);
}

bool MipsSEDAGToDAGISel::selectIntAddr16MM(SDValue Addr, SDValue &Base,
                                         SDValue &Offset) const {
  return selectAddrRegImm16(Addr, Base, Offset) ||
    selectAddrDefault(Addr, Base, Offset);
}

bool MipsSEDAGToDAGISel::selectIntAddrLSL2MM(SDValue Addr, SDValue &Base,
                                             SDValue &Offset) const {
  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 7)) {
    if (isa<FrameIndexSDNode>(Base))
      return false;

    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Offset)) {
      unsigned CnstOff = CN->getZExtValue();
      return (CnstOff == (CnstOff & 0x3c));
    }

    return false;
  }

  // For all other cases where "lw" would be selected, don't select "lw16"
  // because it would result in additional instructions to prepare operands.
  if (selectAddrRegImm(Addr, Base, Offset))
    return false;

  return selectAddrDefault(Addr, Base, Offset);
}

bool MipsSEDAGToDAGISel::selectIntAddrSImm10(SDValue Addr, SDValue &Base,
                                             SDValue &Offset) const {

  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 10))
    return true;

  return selectAddrDefault(Addr, Base, Offset);
}

bool MipsSEDAGToDAGISel::selectIntAddrSImm10Lsl1(SDValue Addr, SDValue &Base,
                                                 SDValue &Offset) const {
  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 10, 1))
    return true;

  return selectAddrDefault(Addr, Base, Offset);
}

bool MipsSEDAGToDAGISel::selectIntAddrSImm10Lsl2(SDValue Addr, SDValue &Base,
                                                 SDValue &Offset) const {
  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 10, 2))
    return true;

  return selectAddrDefault(Addr, Base, Offset);
}

bool MipsSEDAGToDAGISel::selectIntAddrSImm10Lsl3(SDValue Addr, SDValue &Base,
                                                 SDValue &Offset) const {
  if (selectAddrFrameIndex(Addr, Base, Offset))
    return true;

  if (selectAddrFrameIndexOffset(Addr, Base, Offset, 10, 3))
    return true;

  return selectAddrDefault(Addr, Base, Offset);
}

// Select constant vector splats.
//
// Returns true and sets Imm if:
// * MSA is enabled
// * N is a ISD::BUILD_VECTOR representing a constant splat
bool MipsSEDAGToDAGISel::selectVSplat(SDNode *N, APInt &Imm,
                                      unsigned MinSizeInBits) const {
  if (!Subtarget->hasMSA())
    return false;

  BuildVectorSDNode *Node = dyn_cast<BuildVectorSDNode>(N);

  if (!Node)
    return false;

  APInt SplatValue, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;

  if (!Node->isConstantSplat(SplatValue, SplatUndef, SplatBitSize, HasAnyUndefs,
                             MinSizeInBits, !Subtarget->isLittle()))
    return false;

  Imm = SplatValue;

  return true;
}

// Select constant vector splats.
//
// In addition to the requirements of selectVSplat(), this function returns
// true and sets Imm if:
// * The splat value is the same width as the elements of the vector
// * The splat value fits in an integer with the specified signed-ness and
//   width.
//
// This function looks through ISD::BITCAST nodes.
// TODO: This might not be appropriate for big-endian MSA since BITCAST is
//       sometimes a shuffle in big-endian mode.
//
// It's worth noting that this function is not used as part of the selection
// of ldi.[bhwd] since it does not permit using the wrong-typed ldi.[bhwd]
// instruction to achieve the desired bit pattern. ldi.[bhwd] is selected in
// MipsSEDAGToDAGISel::selectNode.
bool MipsSEDAGToDAGISel::
selectVSplatCommon(SDValue N, SDValue &Imm, bool Signed,
                   unsigned ImmBitSize) const {
  APInt ImmValue;
  EVT EltTy = N->getValueType(0).getVectorElementType();

  if (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0);

  if (selectVSplat(N.getNode(), ImmValue, EltTy.getSizeInBits()) &&
      ImmValue.getBitWidth() == EltTy.getSizeInBits()) {

    if (( Signed && ImmValue.isSignedIntN(ImmBitSize)) ||
        (!Signed && ImmValue.isIntN(ImmBitSize))) {
      Imm = CurDAG->getTargetConstant(ImmValue, SDLoc(N), EltTy);
      return true;
    }
  }

  return false;
}

// Select constant vector splats.
bool MipsSEDAGToDAGISel::
selectVSplatUimm1(SDValue N, SDValue &Imm) const {
  return selectVSplatCommon(N, Imm, false, 1);
}

bool MipsSEDAGToDAGISel::
selectVSplatUimm2(SDValue N, SDValue &Imm) const {
  return selectVSplatCommon(N, Imm, false, 2);
}

bool MipsSEDAGToDAGISel::
selectVSplatUimm3(SDValue N, SDValue &Imm) const {
  return selectVSplatCommon(N, Imm, false, 3);
}

// Select constant vector splats.
bool MipsSEDAGToDAGISel::
selectVSplatUimm4(SDValue N, SDValue &Imm) const {
  return selectVSplatCommon(N, Imm, false, 4);
}

// Select constant vector splats.
bool MipsSEDAGToDAGISel::
selectVSplatUimm5(SDValue N, SDValue &Imm) const {
  return selectVSplatCommon(N, Imm, false, 5);
}

// Select constant vector splats.
bool MipsSEDAGToDAGISel::
selectVSplatUimm6(SDValue N, SDValue &Imm) const {
  return selectVSplatCommon(N, Imm, false, 6);
}

// Select constant vector splats.
bool MipsSEDAGToDAGISel::
selectVSplatUimm8(SDValue N, SDValue &Imm) const {
  return selectVSplatCommon(N, Imm, false, 8);
}

// Select constant vector splats.
bool MipsSEDAGToDAGISel::
selectVSplatSimm5(SDValue N, SDValue &Imm) const {
  return selectVSplatCommon(N, Imm, true, 5);
}

// Select constant vector splats whose value is a power of 2.
//
// In addition to the requirements of selectVSplat(), this function returns
// true and sets Imm if:
// * The splat value is the same width as the elements of the vector
// * The splat value is a power of two.
//
// This function looks through ISD::BITCAST nodes.
// TODO: This might not be appropriate for big-endian MSA since BITCAST is
//       sometimes a shuffle in big-endian mode.
bool MipsSEDAGToDAGISel::selectVSplatUimmPow2(SDValue N, SDValue &Imm) const {
  APInt ImmValue;
  EVT EltTy = N->getValueType(0).getVectorElementType();

  if (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0);

  if (selectVSplat(N.getNode(), ImmValue, EltTy.getSizeInBits()) &&
      ImmValue.getBitWidth() == EltTy.getSizeInBits()) {
    int32_t Log2 = ImmValue.exactLogBase2();

    if (Log2 != -1) {
      Imm = CurDAG->getTargetConstant(Log2, SDLoc(N), EltTy);
      return true;
    }
  }

  return false;
}

// Select constant vector splats whose value only has a consecutive sequence
// of left-most bits set (e.g. 0b11...1100...00).
//
// In addition to the requirements of selectVSplat(), this function returns
// true and sets Imm if:
// * The splat value is the same width as the elements of the vector
// * The splat value is a consecutive sequence of left-most bits.
//
// This function looks through ISD::BITCAST nodes.
// TODO: This might not be appropriate for big-endian MSA since BITCAST is
//       sometimes a shuffle in big-endian mode.
bool MipsSEDAGToDAGISel::selectVSplatMaskL(SDValue N, SDValue &Imm) const {
  APInt ImmValue;
  EVT EltTy = N->getValueType(0).getVectorElementType();

  if (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0);

  if (selectVSplat(N.getNode(), ImmValue, EltTy.getSizeInBits()) &&
      ImmValue.getBitWidth() == EltTy.getSizeInBits()) {
    // Extract the run of set bits starting with bit zero from the bitwise
    // inverse of ImmValue, and test that the inverse of this is the same
    // as the original value.
    if (ImmValue == ~(~ImmValue & ~(~ImmValue + 1))) {

      Imm = CurDAG->getTargetConstant(ImmValue.countPopulation(), SDLoc(N),
                                      EltTy);
      return true;
    }
  }

  return false;
}

// Select constant vector splats whose value only has a consecutive sequence
// of right-most bits set (e.g. 0b00...0011...11).
//
// In addition to the requirements of selectVSplat(), this function returns
// true and sets Imm if:
// * The splat value is the same width as the elements of the vector
// * The splat value is a consecutive sequence of right-most bits.
//
// This function looks through ISD::BITCAST nodes.
// TODO: This might not be appropriate for big-endian MSA since BITCAST is
//       sometimes a shuffle in big-endian mode.
bool MipsSEDAGToDAGISel::selectVSplatMaskR(SDValue N, SDValue &Imm) const {
  APInt ImmValue;
  EVT EltTy = N->getValueType(0).getVectorElementType();

  if (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0);

  if (selectVSplat(N.getNode(), ImmValue, EltTy.getSizeInBits()) &&
      ImmValue.getBitWidth() == EltTy.getSizeInBits()) {
    // Extract the run of set bits starting with bit zero, and test that the
    // result is the same as the original value
    if (ImmValue == (ImmValue & ~(ImmValue + 1))) {
      Imm = CurDAG->getTargetConstant(ImmValue.countPopulation(), SDLoc(N),
                                      EltTy);
      return true;
    }
  }

  return false;
}

bool MipsSEDAGToDAGISel::selectVSplatUimmInvPow2(SDValue N,
                                                 SDValue &Imm) const {
  APInt ImmValue;
  EVT EltTy = N->getValueType(0).getVectorElementType();

  if (N->getOpcode() == ISD::BITCAST)
    N = N->getOperand(0);

  if (selectVSplat(N.getNode(), ImmValue, EltTy.getSizeInBits()) &&
      ImmValue.getBitWidth() == EltTy.getSizeInBits()) {
    int32_t Log2 = (~ImmValue).exactLogBase2();

    if (Log2 != -1) {
      Imm = CurDAG->getTargetConstant(Log2, SDLoc(N), EltTy);
      return true;
    }
  }

  return false;
}

bool MipsSEDAGToDAGISel::trySelect(SDNode *Node) {
  unsigned Opcode = Node->getOpcode();
  SDLoc DL(Node);

  ///
  // Instruction Selection not handled by the auto-generated
  // tablegen selection should be handled here.
  ///
  switch(Opcode) {
  default: break;

  case ISD::SUBE: {
    SDValue InFlag = Node->getOperand(2);
    unsigned Opc = Subtarget->isGP64bit() ? Mips::DSUBu : Mips::SUBu;
    selectAddESubE(Opc, InFlag, InFlag.getOperand(0), DL, Node);
    return true;
  }

  case ISD::ADDE: {
    if (Subtarget->hasDSP()) // Select DSP instructions, ADDSC and ADDWC.
      break;
    SDValue InFlag = Node->getOperand(2);
    unsigned Opc = Subtarget->isGP64bit() ? Mips::DADDu : Mips::ADDu;
    selectAddESubE(Opc, InFlag, InFlag.getValue(0), DL, Node);
    return true;
  }

  case ISD::ConstantFP: {
    ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(Node);
    if (Node->getValueType(0) == MVT::f64 && CN->isExactlyValue(+0.0)) {
      if (Subtarget->isGP64bit()) {
        SDValue Zero = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL,
                                              Mips::ZERO_64, MVT::i64);
        ReplaceNode(Node,
                    CurDAG->getMachineNode(Mips::DMTC1, DL, MVT::f64, Zero));
      } else if (Subtarget->isFP64bit()) {
        SDValue Zero = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL,
                                              Mips::ZERO, MVT::i32);
        ReplaceNode(Node, CurDAG->getMachineNode(Mips::BuildPairF64_64, DL,
                                                 MVT::f64, Zero, Zero));
      } else {
        SDValue Zero = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), DL,
                                              Mips::ZERO, MVT::i32);
        ReplaceNode(Node, CurDAG->getMachineNode(Mips::BuildPairF64, DL,
                                                 MVT::f64, Zero, Zero));
      }
      return true;
    }
    break;
  }

  case ISD::Constant: {
    const ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Node);
    int64_t Imm = CN->getSExtValue();
    unsigned Size = CN->getValueSizeInBits(0);

    if (isInt<32>(Imm))
      break;

    MipsAnalyzeImmediate AnalyzeImm;

    const MipsAnalyzeImmediate::InstSeq &Seq =
      AnalyzeImm.Analyze(Imm, Size, false);

    MipsAnalyzeImmediate::InstSeq::const_iterator Inst = Seq.begin();
    SDLoc DL(CN);
    SDNode *RegOpnd;
    SDValue ImmOpnd = CurDAG->getTargetConstant(SignExtend64<16>(Inst->ImmOpnd),
                                                DL, MVT::i64);

    // The first instruction can be a LUi which is different from other
    // instructions (ADDiu, ORI and SLL) in that it does not have a register
    // operand.
    if (Inst->Opc == Mips::LUi64)
      RegOpnd = CurDAG->getMachineNode(Inst->Opc, DL, MVT::i64, ImmOpnd);
    else
      RegOpnd =
        CurDAG->getMachineNode(Inst->Opc, DL, MVT::i64,
                               CurDAG->getRegister(Mips::ZERO_64, MVT::i64),
                               ImmOpnd);

    // The remaining instructions in the sequence are handled here.
    for (++Inst; Inst != Seq.end(); ++Inst) {
      ImmOpnd = CurDAG->getTargetConstant(SignExtend64<16>(Inst->ImmOpnd), DL,
                                          MVT::i64);
      RegOpnd = CurDAG->getMachineNode(Inst->Opc, DL, MVT::i64,
                                       SDValue(RegOpnd, 0), ImmOpnd);
    }

    ReplaceNode(Node, RegOpnd);
    return true;
  }

  case ISD::INTRINSIC_W_CHAIN: {
    switch (cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue()) {
    default:
      break;

    case Intrinsic::mips_cfcmsa: {
      SDValue ChainIn = Node->getOperand(0);
      SDValue RegIdx = Node->getOperand(2);
      SDValue Reg = CurDAG->getCopyFromReg(ChainIn, DL,
                                           getMSACtrlReg(RegIdx), MVT::i32);
      ReplaceNode(Node, Reg.getNode());
      return true;
    }
    }
    break;
  }

  case ISD::INTRINSIC_WO_CHAIN: {
    switch (cast<ConstantSDNode>(Node->getOperand(0))->getZExtValue()) {
    default:
      break;

    case Intrinsic::mips_move_v:
      // Like an assignment but will always produce a move.v even if
      // unnecessary.
      ReplaceNode(Node, CurDAG->getMachineNode(Mips::MOVE_V, DL,
                                               Node->getValueType(0),
                                               Node->getOperand(1)));
      return true;
    }
    break;
  }

  case ISD::INTRINSIC_VOID: {
    switch (cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue()) {
    default:
      break;

    case Intrinsic::mips_ctcmsa: {
      SDValue ChainIn = Node->getOperand(0);
      SDValue RegIdx  = Node->getOperand(2);
      SDValue Value   = Node->getOperand(3);
      SDValue ChainOut = CurDAG->getCopyToReg(ChainIn, DL,
                                              getMSACtrlReg(RegIdx), Value);
      ReplaceNode(Node, ChainOut.getNode());
      return true;
    }
    }
    break;
  }

  case MipsISD::ThreadPointer: {
    EVT PtrVT = getTargetLowering()->getPointerTy(CurDAG->getDataLayout());
    unsigned RdhwrOpc, DestReg;

    if (PtrVT == MVT::i32) {
      RdhwrOpc = Mips::RDHWR;
      DestReg = Mips::V1;
    } else {
      RdhwrOpc = Mips::RDHWR64;
      DestReg = Mips::V1_64;
    }

    SDNode *Rdhwr =
      CurDAG->getMachineNode(RdhwrOpc, DL,
                             Node->getValueType(0),
                             CurDAG->getRegister(Mips::HWR29, MVT::i32));
    SDValue Chain = CurDAG->getCopyToReg(CurDAG->getEntryNode(), DL, DestReg,
                                         SDValue(Rdhwr, 0));
    SDValue ResNode = CurDAG->getCopyFromReg(Chain, DL, DestReg, PtrVT);
    ReplaceNode(Node, ResNode.getNode());
    return true;
  }

  case ISD::BUILD_VECTOR: {
    // Select appropriate ldi.[bhwd] instructions for constant splats of
    // 128-bit when MSA is enabled. Fixup any register class mismatches that
    // occur as a result.
    //
    // This allows the compiler to use a wider range of immediates than would
    // otherwise be allowed. If, for example, v4i32 could only use ldi.h then
    // it would not be possible to load { 0x01010101, 0x01010101, 0x01010101,
    // 0x01010101 } without using a constant pool. This would be sub-optimal
    // when // 'ldi.b wd, 1' is capable of producing that bit-pattern in the
    // same set/ of registers. Similarly, ldi.h isn't capable of producing {
    // 0x00000000, 0x00000001, 0x00000000, 0x00000001 } but 'ldi.d wd, 1' can.

    BuildVectorSDNode *BVN = cast<BuildVectorSDNode>(Node);
    APInt SplatValue, SplatUndef;
    unsigned SplatBitSize;
    bool HasAnyUndefs;
    unsigned LdiOp;
    EVT ResVecTy = BVN->getValueType(0);
    EVT ViaVecTy;

    if (!Subtarget->hasMSA() || !BVN->getValueType(0).is128BitVector())
      return false;

    if (!BVN->isConstantSplat(SplatValue, SplatUndef, SplatBitSize,
                              HasAnyUndefs, 8,
                              !Subtarget->isLittle()))
      return false;

    switch (SplatBitSize) {
    default:
      return false;
    case 8:
      LdiOp = Mips::LDI_B;
      ViaVecTy = MVT::v16i8;
      break;
    case 16:
      LdiOp = Mips::LDI_H;
      ViaVecTy = MVT::v8i16;
      break;
    case 32:
      LdiOp = Mips::LDI_W;
      ViaVecTy = MVT::v4i32;
      break;
    case 64:
      LdiOp = Mips::LDI_D;
      ViaVecTy = MVT::v2i64;
      break;
    }

    if (!SplatValue.isSignedIntN(10))
      return false;

    SDValue Imm = CurDAG->getTargetConstant(SplatValue, DL,
                                            ViaVecTy.getVectorElementType());

    SDNode *Res = CurDAG->getMachineNode(LdiOp, DL, ViaVecTy, Imm);

    if (ResVecTy != ViaVecTy) {
      // If LdiOp is writing to a different register class to ResVecTy, then
      // fix it up here. This COPY_TO_REGCLASS should never cause a move.v
      // since the source and destination register sets contain the same
      // registers.
      const TargetLowering *TLI = getTargetLowering();
      MVT ResVecTySimple = ResVecTy.getSimpleVT();
      const TargetRegisterClass *RC = TLI->getRegClassFor(ResVecTySimple);
      Res = CurDAG->getMachineNode(Mips::COPY_TO_REGCLASS, DL,
                                   ResVecTy, SDValue(Res, 0),
                                   CurDAG->getTargetConstant(RC->getID(), DL,
                                                             MVT::i32));
    }

    ReplaceNode(Node, Res);
    return true;
  }

  }

  return false;
}

bool MipsSEDAGToDAGISel::
SelectInlineAsmMemoryOperand(const SDValue &Op, unsigned ConstraintID,
                             std::vector<SDValue> &OutOps) {
  SDValue Base, Offset;

  switch(ConstraintID) {
  default:
    llvm_unreachable("Unexpected asm memory constraint");
  // All memory constraints can at least accept raw pointers.
  case InlineAsm::Constraint_i:
    OutOps.push_back(Op);
    OutOps.push_back(CurDAG->getTargetConstant(0, SDLoc(Op), MVT::i32));
    return false;
  case InlineAsm::Constraint_m:
    if (selectAddrRegImm16(Op, Base, Offset)) {
      OutOps.push_back(Base);
      OutOps.push_back(Offset);
      return false;
    }
    OutOps.push_back(Op);
    OutOps.push_back(CurDAG->getTargetConstant(0, SDLoc(Op), MVT::i32));
    return false;
  case InlineAsm::Constraint_R:
    // The 'R' constraint is supposed to be much more complicated than this.
    // However, it's becoming less useful due to architectural changes and
    // ought to be replaced by other constraints such as 'ZC'.
    // For now, support 9-bit signed offsets which is supportable by all
    // subtargets for all instructions.
    if (selectAddrRegImm9(Op, Base, Offset)) {
      OutOps.push_back(Base);
      OutOps.push_back(Offset);
      return false;
    }
    OutOps.push_back(Op);
    OutOps.push_back(CurDAG->getTargetConstant(0, SDLoc(Op), MVT::i32));
    return false;
  case InlineAsm::Constraint_ZC:
    // ZC matches whatever the pref, ll, and sc instructions can handle for the
    // given subtarget.
    if (Subtarget->inMicroMipsMode()) {
      // On microMIPS, they can handle 12-bit offsets.
      if (selectAddrRegImm12(Op, Base, Offset)) {
        OutOps.push_back(Base);
        OutOps.push_back(Offset);
        return false;
      }
    } else if (Subtarget->hasMips32r6()) {
      // On MIPS32r6/MIPS64r6, they can only handle 9-bit offsets.
      if (selectAddrRegImm9(Op, Base, Offset)) {
        OutOps.push_back(Base);
        OutOps.push_back(Offset);
        return false;
      }
    } else if (selectAddrRegImm16(Op, Base, Offset)) {
      // Prior to MIPS32r6/MIPS64r6, they can handle 16-bit offsets.
      OutOps.push_back(Base);
      OutOps.push_back(Offset);
      return false;
    }
    // In all cases, 0-bit offsets are acceptable.
    OutOps.push_back(Op);
    OutOps.push_back(CurDAG->getTargetConstant(0, SDLoc(Op), MVT::i32));
    return false;
  }
  return true;
}

FunctionPass *llvm::createMipsSEISelDag(MipsTargetMachine &TM,
                                        CodeGenOpt::Level OptLevel) {
  return new MipsSEDAGToDAGISel(TM, OptLevel);
}
