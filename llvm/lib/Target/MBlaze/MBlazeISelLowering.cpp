//===-- MBlazeISelLowering.cpp - MBlaze DAG Lowering Implementation -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that MBlaze uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mblaze-lower"
#include "MBlazeISelLowering.h"
#include "MBlazeMachineFunction.h"
#include "MBlazeTargetMachine.h"
#include "MBlazeTargetObjectFile.h"
#include "MBlazeSubtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static bool CC_MBlaze_AssignReg(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                CCValAssign::LocInfo &LocInfo,
                                ISD::ArgFlagsTy &ArgFlags,
                                CCState &State);

const char *MBlazeTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
    case MBlazeISD::JmpLink    : return "MBlazeISD::JmpLink";
    case MBlazeISD::GPRel      : return "MBlazeISD::GPRel";
    case MBlazeISD::Wrap       : return "MBlazeISD::Wrap";
    case MBlazeISD::ICmp       : return "MBlazeISD::ICmp";
    case MBlazeISD::Ret        : return "MBlazeISD::Ret";
    case MBlazeISD::Select_CC  : return "MBlazeISD::Select_CC";
    default                    : return NULL;
  }
}

MBlazeTargetLowering::MBlazeTargetLowering(MBlazeTargetMachine &TM)
  : TargetLowering(TM, new MBlazeTargetObjectFile()) {
  Subtarget = &TM.getSubtarget<MBlazeSubtarget>();

  // MBlaze does not have i1 type, so use i32 for
  // setcc operations results (slt, sgt, ...).
  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent); // FIXME: Is this correct?

  // Set up the register classes
  addRegisterClass(MVT::i32, &MBlaze::GPRRegClass);
  if (Subtarget->hasFPU()) {
    addRegisterClass(MVT::f32, &MBlaze::GPRRegClass);
    setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  }

  // Floating point operations which are not supported
  setOperationAction(ISD::FREM,       MVT::f32, Expand);
  setOperationAction(ISD::FMA,        MVT::f32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i8,  Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i16, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  setOperationAction(ISD::FP_ROUND,   MVT::f32, Expand);
  setOperationAction(ISD::FP_ROUND,   MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN,  MVT::f32, Expand);
  setOperationAction(ISD::FCOPYSIGN,  MVT::f64, Expand);
  setOperationAction(ISD::FSIN,       MVT::f32, Expand);
  setOperationAction(ISD::FCOS,       MVT::f32, Expand);
  setOperationAction(ISD::FPOWI,      MVT::f32, Expand);
  setOperationAction(ISD::FPOW,       MVT::f32, Expand);
  setOperationAction(ISD::FLOG,       MVT::f32, Expand);
  setOperationAction(ISD::FLOG2,      MVT::f32, Expand);
  setOperationAction(ISD::FLOG10,     MVT::f32, Expand);
  setOperationAction(ISD::FEXP,       MVT::f32, Expand);

  // Load extented operations for i1 types must be promoted
  setLoadExtAction(ISD::EXTLOAD,  MVT::i1,  Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1,  Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1,  Promote);

  // Sign extended loads must be expanded
  setLoadExtAction(ISD::SEXTLOAD, MVT::i8, Expand);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i16, Expand);

  // MBlaze has no REM or DIVREM operations.
  setOperationAction(ISD::UREM,    MVT::i32, Expand);
  setOperationAction(ISD::SREM,    MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  // If the processor doesn't support multiply then expand it
  if (!Subtarget->hasMul()) {
    setOperationAction(ISD::MUL, MVT::i32, Expand);
  }

  // If the processor doesn't support 64-bit multiply then expand
  if (!Subtarget->hasMul() || !Subtarget->hasMul64()) {
    setOperationAction(ISD::MULHS, MVT::i32, Expand);
    setOperationAction(ISD::MULHS, MVT::i64, Expand);
    setOperationAction(ISD::MULHU, MVT::i32, Expand);
    setOperationAction(ISD::MULHU, MVT::i64, Expand);
  }

  // If the processor doesn't support division then expand
  if (!Subtarget->hasDiv()) {
    setOperationAction(ISD::UDIV, MVT::i32, Expand);
    setOperationAction(ISD::SDIV, MVT::i32, Expand);
  }

  // Expand unsupported conversions
  setOperationAction(ISD::BITCAST, MVT::f32, Expand);
  setOperationAction(ISD::BITCAST, MVT::i32, Expand);

  // Expand SELECT_CC
  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);

  // MBlaze doesn't have MUL_LOHI
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);

  // Used by legalize types to correctly generate the setcc result.
  // Without this, every float setcc comes with a AND/OR with the result,
  // we don't want this, since the fpcmp result goes to a flag register,
  // which is used implicitly by brcond and select operations.
  AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);
  AddPromotedToType(ISD::SELECT, MVT::i1, MVT::i32);
  AddPromotedToType(ISD::SELECT_CC, MVT::i1, MVT::i32);

  // MBlaze Custom Operations
  setOperationAction(ISD::GlobalAddress,      MVT::i32,   Custom);
  setOperationAction(ISD::GlobalTLSAddress,   MVT::i32,   Custom);
  setOperationAction(ISD::JumpTable,          MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,       MVT::i32,   Custom);

  // Variable Argument support
  setOperationAction(ISD::VASTART,            MVT::Other, Custom);
  setOperationAction(ISD::VAEND,              MVT::Other, Expand);
  setOperationAction(ISD::VAARG,              MVT::Other, Expand);
  setOperationAction(ISD::VACOPY,             MVT::Other, Expand);


  // Operations not directly supported by MBlaze.
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32,   Expand);
  setOperationAction(ISD::BR_JT,              MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,              MVT::Other, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG,  MVT::i1,    Expand);
  setOperationAction(ISD::ROTL,               MVT::i32,   Expand);
  setOperationAction(ISD::ROTR,               MVT::i32,   Expand);
  setOperationAction(ISD::SHL_PARTS,          MVT::i32,   Expand);
  setOperationAction(ISD::SRA_PARTS,          MVT::i32,   Expand);
  setOperationAction(ISD::SRL_PARTS,          MVT::i32,   Expand);
  setOperationAction(ISD::CTLZ,               MVT::i32,   Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF,    MVT::i32,   Expand);
  setOperationAction(ISD::CTTZ,               MVT::i32,   Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF,    MVT::i32,   Expand);
  setOperationAction(ISD::CTPOP,              MVT::i32,   Expand);
  setOperationAction(ISD::BSWAP,              MVT::i32,   Expand);

  // We don't have line number support yet.
  setOperationAction(ISD::EH_LABEL,          MVT::Other, Expand);

  // Use the default for now
  setOperationAction(ISD::STACKSAVE,         MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE,      MVT::Other, Expand);

  // MBlaze doesn't have extending float->double load/store
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  setMinFunctionAlignment(2);

  setStackPointerRegisterToSaveRestore(MBlaze::R1);
  computeRegisterProperties();
}

EVT MBlazeTargetLowering::getSetCCResultType(EVT VT) const {
  return MVT::i32;
}

SDValue MBlazeTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode())
  {
    case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
    case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
    case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
    case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
    case ISD::SELECT_CC:          return LowerSELECT_CC(Op, DAG);
    case ISD::VASTART:            return LowerVASTART(Op, DAG);
  }
  return SDValue();
}

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//
MachineBasicBlock*
MBlazeTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                  MachineBasicBlock *MBB)
                                                  const {
  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unexpected instr type to insert");

  case MBlaze::ShiftRL:
  case MBlaze::ShiftRA:
  case MBlaze::ShiftL:
    return EmitCustomShift(MI, MBB);

  case MBlaze::Select_FCC:
  case MBlaze::Select_CC:
    return EmitCustomSelect(MI, MBB);

  case MBlaze::CAS32:
  case MBlaze::SWP32:
  case MBlaze::LAA32:
  case MBlaze::LAS32:
  case MBlaze::LAD32:
  case MBlaze::LAO32:
  case MBlaze::LAX32:
  case MBlaze::LAN32:
    return EmitCustomAtomic(MI, MBB);

  case MBlaze::MEMBARRIER:
    // The Microblaze does not need memory barriers. Just delete the pseudo
    // instruction and finish.
    MI->eraseFromParent();
    return MBB;
  }
}

MachineBasicBlock*
MBlazeTargetLowering::EmitCustomShift(MachineInstr *MI,
                                      MachineBasicBlock *MBB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();

  // To "insert" a shift left instruction, we actually have to insert a
  // simple loop.  The incoming instruction knows the destination vreg to
  // set, the source vreg to operate over and the shift amount.
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator It = MBB;
  ++It;

  // start:
  //   andi     samt, samt, 31
  //   beqid    samt, finish
  //   add      dst, src, r0
  // loop:
  //   addik    samt, samt, -1
  //   sra      dst, dst
  //   bneid    samt, loop
  //   nop
  // finish:
  MachineFunction *F = MBB->getParent();
  MachineRegisterInfo &R = F->getRegInfo();
  MachineBasicBlock *loop = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *finish = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(It, loop);
  F->insert(It, finish);

  // Update machine-CFG edges by transferring adding all successors and
  // remaining instructions from the current block to the new block which
  // will contain the Phi node for the select.
  finish->splice(finish->begin(), MBB,
                 llvm::next(MachineBasicBlock::iterator(MI)),
                 MBB->end());
  finish->transferSuccessorsAndUpdatePHIs(MBB);

  // Add the true and fallthrough blocks as its successors.
  MBB->addSuccessor(loop);
  MBB->addSuccessor(finish);

  // Next, add the finish block as a successor of the loop block
  loop->addSuccessor(finish);
  loop->addSuccessor(loop);

  unsigned IAMT = R.createVirtualRegister(&MBlaze::GPRRegClass);
  BuildMI(MBB, dl, TII->get(MBlaze::ANDI), IAMT)
    .addReg(MI->getOperand(2).getReg())
    .addImm(31);

  unsigned IVAL = R.createVirtualRegister(&MBlaze::GPRRegClass);
  BuildMI(MBB, dl, TII->get(MBlaze::ADDIK), IVAL)
    .addReg(MI->getOperand(1).getReg())
    .addImm(0);

  BuildMI(MBB, dl, TII->get(MBlaze::BEQID))
    .addReg(IAMT)
    .addMBB(finish);

  unsigned DST = R.createVirtualRegister(&MBlaze::GPRRegClass);
  unsigned NDST = R.createVirtualRegister(&MBlaze::GPRRegClass);
  BuildMI(loop, dl, TII->get(MBlaze::PHI), DST)
    .addReg(IVAL).addMBB(MBB)
    .addReg(NDST).addMBB(loop);

  unsigned SAMT = R.createVirtualRegister(&MBlaze::GPRRegClass);
  unsigned NAMT = R.createVirtualRegister(&MBlaze::GPRRegClass);
  BuildMI(loop, dl, TII->get(MBlaze::PHI), SAMT)
    .addReg(IAMT).addMBB(MBB)
    .addReg(NAMT).addMBB(loop);

  if (MI->getOpcode() == MBlaze::ShiftL)
    BuildMI(loop, dl, TII->get(MBlaze::ADD), NDST).addReg(DST).addReg(DST);
  else if (MI->getOpcode() == MBlaze::ShiftRA)
    BuildMI(loop, dl, TII->get(MBlaze::SRA), NDST).addReg(DST);
  else if (MI->getOpcode() == MBlaze::ShiftRL)
    BuildMI(loop, dl, TII->get(MBlaze::SRL), NDST).addReg(DST);
  else
    llvm_unreachable("Cannot lower unknown shift instruction");

  BuildMI(loop, dl, TII->get(MBlaze::ADDIK), NAMT)
    .addReg(SAMT)
    .addImm(-1);

  BuildMI(loop, dl, TII->get(MBlaze::BNEID))
    .addReg(NAMT)
    .addMBB(loop);

  BuildMI(*finish, finish->begin(), dl,
          TII->get(MBlaze::PHI), MI->getOperand(0).getReg())
    .addReg(IVAL).addMBB(MBB)
    .addReg(NDST).addMBB(loop);

  // The pseudo instruction is no longer needed so remove it
  MI->eraseFromParent();
  return finish;
}

MachineBasicBlock*
MBlazeTargetLowering::EmitCustomSelect(MachineInstr *MI,
                                       MachineBasicBlock *MBB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();

  // To "insert" a SELECT_CC instruction, we actually have to insert the
  // diamond control-flow pattern.  The incoming instruction knows the
  // destination vreg to set, the condition code register to branch on, the
  // true/false values to select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator It = MBB;
  ++It;

  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   setcc r1, r2, r3
  //   bNE   r1, r0, copy1MBB
  //   fallthrough --> copy0MBB
  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *flsBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *dneBB = F->CreateMachineBasicBlock(LLVM_BB);

  unsigned Opc;
  switch (MI->getOperand(4).getImm()) {
  default: llvm_unreachable("Unknown branch condition");
  case MBlazeCC::EQ: Opc = MBlaze::BEQID; break;
  case MBlazeCC::NE: Opc = MBlaze::BNEID; break;
  case MBlazeCC::GT: Opc = MBlaze::BGTID; break;
  case MBlazeCC::LT: Opc = MBlaze::BLTID; break;
  case MBlazeCC::GE: Opc = MBlaze::BGEID; break;
  case MBlazeCC::LE: Opc = MBlaze::BLEID; break;
  }

  F->insert(It, flsBB);
  F->insert(It, dneBB);

  // Transfer the remainder of MBB and its successor edges to dneBB.
  dneBB->splice(dneBB->begin(), MBB,
                llvm::next(MachineBasicBlock::iterator(MI)),
                MBB->end());
  dneBB->transferSuccessorsAndUpdatePHIs(MBB);

  MBB->addSuccessor(flsBB);
  MBB->addSuccessor(dneBB);
  flsBB->addSuccessor(dneBB);

  BuildMI(MBB, dl, TII->get(Opc))
    .addReg(MI->getOperand(3).getReg())
    .addMBB(dneBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  //BuildMI(dneBB, dl, TII->get(MBlaze::PHI), MI->getOperand(0).getReg())
  //  .addReg(MI->getOperand(1).getReg()).addMBB(flsBB)
  //  .addReg(MI->getOperand(2).getReg()).addMBB(BB);

  BuildMI(*dneBB, dneBB->begin(), dl,
          TII->get(MBlaze::PHI), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(flsBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(MBB);

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return dneBB;
}

MachineBasicBlock*
MBlazeTargetLowering::EmitCustomAtomic(MachineInstr *MI,
                                       MachineBasicBlock *MBB) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();

  // All atomic instructions on the Microblaze are implemented using the
  // load-linked / store-conditional style atomic instruction sequences.
  // Thus, all operations will look something like the following:
  //
  //  start:
  //    lwx     RV, RP, 0
  //    <do stuff>
  //    swx     RV, RP, 0
  //    addic   RC, R0, 0
  //    bneid   RC, start
  //
  //  exit:
  //
  // To "insert" a shift left instruction, we actually have to insert a
  // simple loop.  The incoming instruction knows the destination vreg to
  // set, the source vreg to operate over and the shift amount.
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator It = MBB;
  ++It;

  // start:
  //   andi     samt, samt, 31
  //   beqid    samt, finish
  //   add      dst, src, r0
  // loop:
  //   addik    samt, samt, -1
  //   sra      dst, dst
  //   bneid    samt, loop
  //   nop
  // finish:
  MachineFunction *F = MBB->getParent();
  MachineRegisterInfo &R = F->getRegInfo();

  // Create the start and exit basic blocks for the atomic operation
  MachineBasicBlock *start = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exit = F->CreateMachineBasicBlock(LLVM_BB);
  F->insert(It, start);
  F->insert(It, exit);

  // Update machine-CFG edges by transferring adding all successors and
  // remaining instructions from the current block to the new block which
  // will contain the Phi node for the select.
  exit->splice(exit->begin(), MBB, llvm::next(MachineBasicBlock::iterator(MI)),
               MBB->end());
  exit->transferSuccessorsAndUpdatePHIs(MBB);

  // Add the fallthrough block as its successors.
  MBB->addSuccessor(start);

  BuildMI(start, dl, TII->get(MBlaze::LWX), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(1).getReg())
    .addReg(MBlaze::R0);

  MachineBasicBlock *final = start;
  unsigned finalReg = 0;

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Cannot lower unknown atomic instruction!");

  case MBlaze::SWP32:
    finalReg = MI->getOperand(2).getReg();
    start->addSuccessor(exit);
    start->addSuccessor(start);
    break;

  case MBlaze::LAN32:
  case MBlaze::LAX32:
  case MBlaze::LAO32:
  case MBlaze::LAD32:
  case MBlaze::LAS32:
  case MBlaze::LAA32: {
    unsigned opcode = 0;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("Cannot lower unknown atomic load!");
    case MBlaze::LAA32: opcode = MBlaze::ADDIK; break;
    case MBlaze::LAS32: opcode = MBlaze::RSUBIK; break;
    case MBlaze::LAD32: opcode = MBlaze::AND; break;
    case MBlaze::LAO32: opcode = MBlaze::OR; break;
    case MBlaze::LAX32: opcode = MBlaze::XOR; break;
    case MBlaze::LAN32: opcode = MBlaze::AND; break;
    }

    finalReg = R.createVirtualRegister(&MBlaze::GPRRegClass);
    start->addSuccessor(exit);
    start->addSuccessor(start);

    BuildMI(start, dl, TII->get(opcode), finalReg)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(2).getReg());

    if (MI->getOpcode() == MBlaze::LAN32) {
      unsigned tmp = finalReg;
      finalReg = R.createVirtualRegister(&MBlaze::GPRRegClass);
      BuildMI(start, dl, TII->get(MBlaze::XORI), finalReg)
        .addReg(tmp)
        .addImm(-1);
    }
    break;
  }

  case MBlaze::CAS32: {
    finalReg = MI->getOperand(3).getReg();
    final = F->CreateMachineBasicBlock(LLVM_BB);

    F->insert(It, final);
    start->addSuccessor(exit);
    start->addSuccessor(final);
    final->addSuccessor(exit);
    final->addSuccessor(start);

    unsigned CMP = R.createVirtualRegister(&MBlaze::GPRRegClass);
    BuildMI(start, dl, TII->get(MBlaze::CMP), CMP)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(2).getReg());

    BuildMI(start, dl, TII->get(MBlaze::BNEID))
      .addReg(CMP)
      .addMBB(exit);

    final->moveAfter(start);
    exit->moveAfter(final);
    break;
  }
  }

  unsigned CHK = R.createVirtualRegister(&MBlaze::GPRRegClass);
  BuildMI(final, dl, TII->get(MBlaze::SWX))
    .addReg(finalReg)
    .addReg(MI->getOperand(1).getReg())
    .addReg(MBlaze::R0);

  BuildMI(final, dl, TII->get(MBlaze::ADDIC), CHK)
    .addReg(MBlaze::R0)
    .addImm(0);

  BuildMI(final, dl, TII->get(MBlaze::BNEID))
    .addReg(CHK)
    .addMBB(start);

  // The pseudo instruction is no longer needed so remove it
  MI->eraseFromParent();
  return exit;
}

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//
//

SDValue MBlazeTargetLowering::LowerSELECT_CC(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);
  DebugLoc dl = Op.getDebugLoc();
  unsigned Opc;

  SDValue CompareFlag;
  if (LHS.getValueType() == MVT::i32) {
    Opc = MBlazeISD::Select_CC;
    CompareFlag = DAG.getNode(MBlazeISD::ICmp, dl, MVT::i32, LHS, RHS)
                    .getValue(1);
  } else {
    llvm_unreachable("Cannot lower select_cc with unknown type");
  }

  return DAG.getNode(Opc, dl, TrueVal.getValueType(), TrueVal, FalseVal,
                     CompareFlag);
}

SDValue MBlazeTargetLowering::
LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const {
  // FIXME there isn't actually debug info here
  DebugLoc dl = Op.getDebugLoc();
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDValue GA = DAG.getTargetGlobalAddress(GV, dl, MVT::i32);

  return DAG.getNode(MBlazeISD::Wrap, dl, MVT::i32, GA);
}

SDValue MBlazeTargetLowering::
LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const {
  llvm_unreachable("TLS not implemented for MicroBlaze.");
}

SDValue MBlazeTargetLowering::
LowerJumpTable(SDValue Op, SelectionDAG &DAG) const {
  SDValue ResNode;
  SDValue HiPart;
  // FIXME there isn't actually debug info here
  DebugLoc dl = Op.getDebugLoc();

  EVT PtrVT = Op.getValueType();
  JumpTableSDNode *JT  = cast<JumpTableSDNode>(Op);

  SDValue JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT, 0);
  return DAG.getNode(MBlazeISD::Wrap, dl, MVT::i32, JTI);
}

SDValue MBlazeTargetLowering::
LowerConstantPool(SDValue Op, SelectionDAG &DAG) const {
  SDValue ResNode;
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);
  const Constant *C = N->getConstVal();
  DebugLoc dl = Op.getDebugLoc();

  SDValue CP = DAG.getTargetConstantPool(C, MVT::i32, N->getAlignment(),
                                         N->getOffset(), 0);
  return DAG.getNode(MBlazeISD::Wrap, dl, MVT::i32, CP);
}

SDValue MBlazeTargetLowering::LowerVASTART(SDValue Op,
                                           SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MBlazeFunctionInfo *FuncInfo = MF.getInfo<MBlazeFunctionInfo>();

  DebugLoc dl = Op.getDebugLoc();
  SDValue FI = DAG.getFrameIndex(FuncInfo->getVarArgsFrameIndex(),
                                 getPointerTy());

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), dl, FI, Op.getOperand(1),
                      MachinePointerInfo(SV),
                      false, false, 0);
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "MBlazeGenCallingConv.inc"

static bool CC_MBlaze_AssignReg(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                CCValAssign::LocInfo &LocInfo,
                                ISD::ArgFlagsTy &ArgFlags,
                                CCState &State) {
  static const uint16_t ArgRegs[] = {
    MBlaze::R5, MBlaze::R6, MBlaze::R7,
    MBlaze::R8, MBlaze::R9, MBlaze::R10
  };

  const unsigned NumArgRegs = array_lengthof(ArgRegs);
  unsigned Reg = State.AllocateReg(ArgRegs, NumArgRegs);
  if (!Reg) return false;

  unsigned SizeInBytes = ValVT.getSizeInBits() >> 3;
  State.AllocateStack(SizeInBytes, SizeInBytes);
  State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));

  return true;
}

//===----------------------------------------------------------------------===//
//                  Call Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// LowerCall - functions arguments are copied from virtual regs to
/// (physical regs)/(stack frame), CALLSEQ_START and CALLSEQ_END are emitted.
/// TODO: isVarArg, isTailCall.
SDValue MBlazeTargetLowering::
LowerCall(TargetLowering::CallLoweringInfo &CLI,
          SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG                     = CLI.DAG;
  DebugLoc &dl                          = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals     = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins   = CLI.Ins;
  SDValue Chain                         = CLI.Chain;
  SDValue Callee                        = CLI.Callee;
  bool &isTailCall                      = CLI.IsTailCall;
  CallingConv::ID CallConv              = CLI.CallConv;
  bool isVarArg                         = CLI.IsVarArg;

  // MBlaze does not yet support tail call optimization
  isTailCall = false;

  // The MBlaze requires stack slots for arguments passed to var arg
  // functions even if they are passed in registers.
  bool needsRegArgSlots = isVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetFrameLowering &TFI = *MF.getTarget().getFrameLowering();

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeCallOperands(Outs, CC_MBlaze);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  // Variable argument function calls require a minimum of 24-bytes of stack
  if (isVarArg && NumBytes < 24) NumBytes = 24;

  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    MVT RegVT = VA.getLocVT();
    SDValue Arg = OutVals[i];

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, dl, RegVT, Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, dl, RegVT, Arg);
      break;
    }

    // Arguments that can be passed on register must be kept at
    // RegsToPass vector
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      // Register can't get to this point...
      assert(VA.isMemLoc());

      // Since we are alread passing values on the stack we don't
      // need to worry about creating additional slots for the
      // values passed via registers.
      needsRegArgSlots = false;

      // Create the frame index object for this incoming parameter
      unsigned ArgSize = VA.getValVT().getSizeInBits()/8;
      unsigned StackLoc = VA.getLocMemOffset() + 4;
      int FI = MFI->CreateFixedObject(ArgSize, StackLoc, true);

      SDValue PtrOff = DAG.getFrameIndex(FI,getPointerTy());

      // emit ISD::STORE whichs stores the
      // parameter value to a stack Location
      MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                         MachinePointerInfo(),
                                         false, false, 0));
    }
  }

  // If we need to reserve stack space for the arguments passed via registers
  // then create a fixed stack object at the beginning of the stack.
  if (needsRegArgSlots && TFI.hasReservedCallFrame(MF))
    MFI->CreateFixedObject(28,0,true);

  // Transform all store nodes into one single node because all store
  // nodes are independent of each other.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emitted instructions must be
  // stuck together.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl,
                                getPointerTy(), 0, 0);
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(),
                                getPointerTy(), 0);

  // MBlazeJmpLink = #chain, #target_address, #opt_in_flags...
  //             = Chain, Callee, Reg#1, Reg#2, ...
  //
  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));
  }

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  Chain  = DAG.getNode(MBlazeISD::JmpLink, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg,
                         Ins, dl, DAG, InVals);
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
SDValue MBlazeTargetLowering::
LowerCallResult(SDValue Chain, SDValue InFlag, CallingConv::ID CallConv,
                bool isVarArg, const SmallVectorImpl<ISD::InputArg> &Ins,
                DebugLoc dl, SelectionDAG &DAG,
                SmallVectorImpl<SDValue> &InVals) const {
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  CCInfo.AnalyzeCallResult(Ins, RetCC_MBlaze);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    Chain = DAG.getCopyFromReg(Chain, dl, RVLocs[i].getLocReg(),
                               RVLocs[i].getValVT(), InFlag).getValue(1);
    InFlag = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//             Formal Arguments Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// LowerFormalArguments - transform physical registers into
/// virtual registers and generate load operations for
/// arguments places on the stack.
SDValue MBlazeTargetLowering::
LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
                     const SmallVectorImpl<ISD::InputArg> &Ins,
                     DebugLoc dl, SelectionDAG &DAG,
                     SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MBlazeFunctionInfo *MBlazeFI = MF.getInfo<MBlazeFunctionInfo>();

  unsigned StackReg = MF.getTarget().getRegisterInfo()->getFrameRegister(MF);
  MBlazeFI->setVarArgsFrameIndex(0);

  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Keep track of the last register used for arguments
  unsigned ArgRegEnd = 0;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  CCInfo.AnalyzeFormalArguments(Ins, CC_MBlaze);
  SDValue StackPtr;

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    // Arguments stored on registers
    if (VA.isRegLoc()) {
      MVT RegVT = VA.getLocVT();
      ArgRegEnd = VA.getLocReg();
      const TargetRegisterClass *RC;

      if (RegVT == MVT::i32)
        RC = &MBlaze::GPRRegClass;
      else if (RegVT == MVT::f32)
        RC = &MBlaze::GPRRegClass;
      else
        llvm_unreachable("RegVT not supported by LowerFormalArguments");

      // Transform the arguments stored on
      // physical registers into virtual ones
      unsigned Reg = MF.addLiveIn(ArgRegEnd, RC);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);

      // If this is an 8 or 16-bit value, it has been passed promoted
      // to 32 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size. If if is a floating point value
      // then convert to the correct type.
      if (VA.getLocInfo() != CCValAssign::Full) {
        unsigned Opcode = 0;
        if (VA.getLocInfo() == CCValAssign::SExt)
          Opcode = ISD::AssertSext;
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          Opcode = ISD::AssertZext;
        if (Opcode)
          ArgValue = DAG.getNode(Opcode, dl, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);
      }

      InVals.push_back(ArgValue);
    } else { // VA.isRegLoc()
      // sanity check
      assert(VA.isMemLoc());

      // The last argument is not a register
      ArgRegEnd = 0;

      // The stack pointer offset is relative to the caller stack frame.
      // Since the real stack size is unknown here, a negative SPOffset
      // is used so there's a way to adjust these offsets when the stack
      // size get known (on EliminateFrameIndex). A dummy SPOffset is
      // used instead of a direct negative address (which is recorded to
      // be used on emitPrologue) to avoid mis-calc of the first stack
      // offset on PEI::calculateFrameObjectOffsets.
      // Arguments are always 32-bit.
      unsigned ArgSize = VA.getLocVT().getSizeInBits()/8;
      unsigned StackLoc = VA.getLocMemOffset() + 4;
      int FI = MFI->CreateFixedObject(ArgSize, 0, true);
      MBlazeFI->recordLoadArgsFI(FI, -StackLoc);
      MBlazeFI->recordLiveIn(FI);

      // Create load nodes to retrieve arguments from the stack
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
      InVals.push_back(DAG.getLoad(VA.getValVT(), dl, Chain, FIN,
                                   MachinePointerInfo::getFixedStack(FI),
                                   false, false, false, 0));
    }
  }

  // To meet ABI, when VARARGS are passed on registers, the registers
  // must have their values written to the caller stack frame. If the last
  // argument was placed in the stack, there's no need to save any register.
  if ((isVarArg) && ArgRegEnd) {
    if (StackPtr.getNode() == 0)
      StackPtr = DAG.getRegister(StackReg, getPointerTy());

    // The last register argument that must be saved is MBlaze::R10
    const TargetRegisterClass *RC = &MBlaze::GPRRegClass;

    unsigned Begin = getMBlazeRegisterNumbering(MBlaze::R5);
    unsigned Start = getMBlazeRegisterNumbering(ArgRegEnd+1);
    unsigned End   = getMBlazeRegisterNumbering(MBlaze::R10);
    unsigned StackLoc = Start - Begin + 1;

    for (; Start <= End; ++Start, ++StackLoc) {
      unsigned Reg = getMBlazeRegisterFromNumbering(Start);
      unsigned LiveReg = MF.addLiveIn(Reg, RC);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, dl, LiveReg, MVT::i32);

      int FI = MFI->CreateFixedObject(4, 0, true);
      MBlazeFI->recordStoreVarArgsFI(FI, -(StackLoc*4));
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy());
      OutChains.push_back(DAG.getStore(Chain, dl, ArgValue, PtrOff,
                                       MachinePointerInfo(),
                                       false, false, 0));

      // Record the frame index of the first variable argument
      // which is a value necessary to VASTART.
      if (!MBlazeFI->getVarArgsFrameIndex())
        MBlazeFI->setVarArgsFrameIndex(FI);
    }
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens when on varg functions
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &OutChains[0], OutChains.size());
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

SDValue MBlazeTargetLowering::
LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
            const SmallVectorImpl<ISD::OutputArg> &Outs,
            const SmallVectorImpl<SDValue> &OutVals,
            DebugLoc dl, SelectionDAG &DAG) const {
  // CCValAssign - represent the assignment of
  // the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  // Analize return values.
  CCInfo.AnalyzeReturn(Outs, RetCC_MBlaze);

  // If this is the first return lowered for this function, add
  // the regs to the liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(),
                             OutVals[i], Flag);

    // guarantee that all emitted copies are
    // stuck together, avoiding something bad
    Flag = Chain.getValue(1);
  }

  // If this function is using the interrupt_handler calling convention
  // then use "rtid r14, 0" otherwise use "rtsd r15, 8"
  unsigned Ret = (CallConv == CallingConv::MBLAZE_INTR) ? MBlazeISD::IRet
                                                        : MBlazeISD::Ret;
  unsigned Reg = (CallConv == CallingConv::MBLAZE_INTR) ? MBlaze::R14
                                                        : MBlaze::R15;
  SDValue DReg = DAG.getRegister(Reg, MVT::i32);

  if (Flag.getNode())
    return DAG.getNode(Ret, dl, MVT::Other, Chain, DReg, Flag);

  return DAG.getNode(Ret, dl, MVT::Other, Chain, DReg);
}

//===----------------------------------------------------------------------===//
//                           MBlaze Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
MBlazeTargetLowering::ConstraintType MBlazeTargetLowering::
getConstraintType(const std::string &Constraint) const
{
  // MBlaze specific constrainy
  //
  // 'd' : An address register. Equivalent to r.
  // 'y' : Equivalent to r; retained for
  //       backwards compatibility.
  // 'f' : Floating Point registers.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
      default : break;
      case 'd':
      case 'y':
      case 'f':
        return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
MBlazeTargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
    // If we don't have a value, we can't do a match,
    // but allow it at the lowest weight.
  if (CallOperandVal == NULL)
    return CW_Default;
  Type *type = CallOperandVal->getType();
  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;
  case 'd':
  case 'y':
    if (type->isIntegerTy())
      weight = CW_Register;
    break;
  case 'f':
    if (type->isFloatTy())
      weight = CW_Register;
    break;
  }
  return weight;
}

/// Given a register class constraint, like 'r', if this corresponds directly
/// to an LLVM register class, return a register of 0 and the register class
/// pointer.
std::pair<unsigned, const TargetRegisterClass*> MBlazeTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint, EVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, &MBlaze::GPRRegClass);
      // TODO: These can't possibly be right, but match what was in
      // getRegClassForInlineAsmConstraint.
    case 'd':
    case 'y':
    case 'f':
      if (VT == MVT::f32)
        return std::make_pair(0U, &MBlaze::GPRRegClass);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

bool MBlazeTargetLowering::
isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The MBlaze target isn't yet aware of offsets.
  return false;
}

bool MBlazeTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  return VT != MVT::f32;
}
