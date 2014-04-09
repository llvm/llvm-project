//===-- PPCISelDAGToDAG.cpp - PPC --pattern matching inst selector --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for PowerPC,
// converting from a legalized dag to a PPC dag.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ppc-codegen"
#include "PPC.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

namespace llvm {
  void initializePPCDAGToDAGISelPass(PassRegistry&);
}

namespace {
  //===--------------------------------------------------------------------===//
  /// PPCDAGToDAGISel - PPC specific code to select PPC machine
  /// instructions for SelectionDAG operations.
  ///
  class PPCDAGToDAGISel : public SelectionDAGISel {
    const PPCTargetMachine &TM;
    const PPCTargetLowering &PPCLowering;
    const PPCSubtarget &PPCSubTarget;
    unsigned GlobalBaseReg;
  public:
    explicit PPCDAGToDAGISel(PPCTargetMachine &tm)
      : SelectionDAGISel(tm), TM(tm),
        PPCLowering(*TM.getTargetLowering()),
        PPCSubTarget(*TM.getSubtargetImpl()) {
      initializePPCDAGToDAGISelPass(*PassRegistry::getPassRegistry());
    }

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseReg = 0;
      SelectionDAGISel::runOnMachineFunction(MF);

      if (!PPCSubTarget.isSVR4ABI())
        InsertVRSaveCode(MF);

      return true;
    }

    virtual void PostprocessISelDAG();

    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDValue getI32Imm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }

    /// getI64Imm - Return a target constant with the specified value, of type
    /// i64.
    inline SDValue getI64Imm(uint64_t Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i64);
    }

    /// getSmallIPtrImm - Return a target constant of pointer type.
    inline SDValue getSmallIPtrImm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, PPCLowering.getPointerTy());
    }

    /// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s
    /// with any number of 0s on either side.  The 1s are allowed to wrap from
    /// LSB to MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.
    /// 0x0F0F0000 is not, since all 1s are not contiguous.
    static bool isRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME);


    /// isRotateAndMask - Returns true if Mask and Shift can be folded into a
    /// rotate and mask opcode and mask operation.
    static bool isRotateAndMask(SDNode *N, unsigned Mask, bool isShiftMask,
                                unsigned &SH, unsigned &MB, unsigned &ME);

    /// getGlobalBaseReg - insert code into the entry mbb to materialize the PIC
    /// base register.  Return the virtual register that holds this value.
    SDNode *getGlobalBaseReg();

    // Select - Convert the specified operand from a target-independent to a
    // target-specific node if it hasn't already been changed.
    SDNode *Select(SDNode *N);

    SDNode *SelectBitfieldInsert(SDNode *N);

    /// SelectCC - Select a comparison of the specified values with the
    /// specified condition code, returning the CR# of the expression.
    SDValue SelectCC(SDValue LHS, SDValue RHS, ISD::CondCode CC, SDLoc dl);

    /// SelectAddrImm - Returns true if the address N can be represented by
    /// a base register plus a signed 16-bit displacement [r+imm].
    bool SelectAddrImm(SDValue N, SDValue &Disp,
                       SDValue &Base) {
      return PPCLowering.SelectAddressRegImm(N, Disp, Base, *CurDAG, false);
    }

    /// SelectAddrImmOffs - Return true if the operand is valid for a preinc
    /// immediate field.  Note that the operand at this point is already the
    /// result of a prior SelectAddressRegImm call.
    bool SelectAddrImmOffs(SDValue N, SDValue &Out) const {
      if (N.getOpcode() == ISD::TargetConstant ||
          N.getOpcode() == ISD::TargetGlobalAddress) {
        Out = N;
        return true;
      }

      return false;
    }

    /// SelectAddrIdx - Given the specified addressed, check to see if it can be
    /// represented as an indexed [r+r] operation.  Returns false if it can
    /// be represented by [r+imm], which are preferred.
    bool SelectAddrIdx(SDValue N, SDValue &Base, SDValue &Index) {
      return PPCLowering.SelectAddressRegReg(N, Base, Index, *CurDAG);
    }

    /// SelectAddrIdxOnly - Given the specified addressed, force it to be
    /// represented as an indexed [r+r] operation.
    bool SelectAddrIdxOnly(SDValue N, SDValue &Base, SDValue &Index) {
      return PPCLowering.SelectAddressRegRegOnly(N, Base, Index, *CurDAG);
    }

    /// SelectAddrImmX4 - Returns true if the address N can be represented by
    /// a base register plus a signed 16-bit displacement that is a multiple of 4.
    /// Suitable for use by STD and friends.
    bool SelectAddrImmX4(SDValue N, SDValue &Disp, SDValue &Base) {
      return PPCLowering.SelectAddressRegImm(N, Disp, Base, *CurDAG, true);
    }

    // Select an address into a single register.
    bool SelectAddr(SDValue N, SDValue &Base) {
      Base = N;
      return true;
    }

    /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
    /// inline asm expressions.  It is always correct to compute the value into
    /// a register.  The case of adding a (possibly relocatable) constant to a
    /// register can be improved, but it is wrong to substitute Reg+Reg for
    /// Reg in an asm, because the load or store opcode would have to change.
   virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                              char ConstraintCode,
                                              std::vector<SDValue> &OutOps) {
      OutOps.push_back(Op);
      return false;
    }

    void InsertVRSaveCode(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "PowerPC DAG->DAG Pattern Instruction Selection";
    }

// Include the pieces autogenerated from the target description.
#include "PPCGenDAGISel.inc"

private:
    SDNode *SelectSETCC(SDNode *N);
  };
}

/// InsertVRSaveCode - Once the entire function has been instruction selected,
/// all virtual registers are created and all machine instructions are built,
/// check to see if we need to save/restore VRSAVE.  If so, do it.
void PPCDAGToDAGISel::InsertVRSaveCode(MachineFunction &Fn) {
  // Check to see if this function uses vector registers, which means we have to
  // save and restore the VRSAVE register and update it with the regs we use.
  //
  // In this case, there will be virtual registers of vector type created
  // by the scheduler.  Detect them now.
  bool HasVectorVReg = false;
  for (unsigned i = 0, e = RegInfo->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (RegInfo->getRegClass(Reg) == &PPC::VRRCRegClass) {
      HasVectorVReg = true;
      break;
    }
  }
  if (!HasVectorVReg) return;  // nothing to do.

  // If we have a vector register, we want to emit code into the entry and exit
  // blocks to save and restore the VRSAVE register.  We do this here (instead
  // of marking all vector instructions as clobbering VRSAVE) for two reasons:
  //
  // 1. This (trivially) reduces the load on the register allocator, by not
  //    having to represent the live range of the VRSAVE register.
  // 2. This (more significantly) allows us to create a temporary virtual
  //    register to hold the saved VRSAVE value, allowing this temporary to be
  //    register allocated, instead of forcing it to be spilled to the stack.

  // Create two vregs - one to hold the VRSAVE register that is live-in to the
  // function and one for the value after having bits or'd into it.
  unsigned InVRSAVE = RegInfo->createVirtualRegister(&PPC::GPRCRegClass);
  unsigned UpdatedVRSAVE = RegInfo->createVirtualRegister(&PPC::GPRCRegClass);

  const TargetInstrInfo &TII = *TM.getInstrInfo();
  MachineBasicBlock &EntryBB = *Fn.begin();
  DebugLoc dl;
  // Emit the following code into the entry block:
  // InVRSAVE = MFVRSAVE
  // UpdatedVRSAVE = UPDATE_VRSAVE InVRSAVE
  // MTVRSAVE UpdatedVRSAVE
  MachineBasicBlock::iterator IP = EntryBB.begin();  // Insert Point
  BuildMI(EntryBB, IP, dl, TII.get(PPC::MFVRSAVE), InVRSAVE);
  BuildMI(EntryBB, IP, dl, TII.get(PPC::UPDATE_VRSAVE),
          UpdatedVRSAVE).addReg(InVRSAVE);
  BuildMI(EntryBB, IP, dl, TII.get(PPC::MTVRSAVE)).addReg(UpdatedVRSAVE);

  // Find all return blocks, outputting a restore in each epilog.
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    if (!BB->empty() && BB->back().isReturn()) {
      IP = BB->end(); --IP;

      // Skip over all terminator instructions, which are part of the return
      // sequence.
      MachineBasicBlock::iterator I2 = IP;
      while (I2 != BB->begin() && (--I2)->isTerminator())
        IP = I2;

      // Emit: MTVRSAVE InVRSave
      BuildMI(*BB, IP, dl, TII.get(PPC::MTVRSAVE)).addReg(InVRSAVE);
    }
  }
}


/// getGlobalBaseReg - Output the instructions required to put the
/// base address to use for accessing globals into a register.
///
SDNode *PPCDAGToDAGISel::getGlobalBaseReg() {
  if (!GlobalBaseReg) {
    const TargetInstrInfo &TII = *TM.getInstrInfo();
    // Insert the set of GlobalBaseReg into the first MBB of the function
    MachineBasicBlock &FirstMBB = MF->front();
    MachineBasicBlock::iterator MBBI = FirstMBB.begin();
    DebugLoc dl;

    if (PPCLowering.getPointerTy() == MVT::i32) {
      GlobalBaseReg = RegInfo->createVirtualRegister(&PPC::GPRC_NOR0RegClass);
      BuildMI(FirstMBB, MBBI, dl, TII.get(PPC::MovePCtoLR));
      BuildMI(FirstMBB, MBBI, dl, TII.get(PPC::MFLR), GlobalBaseReg);
    } else {
      GlobalBaseReg = RegInfo->createVirtualRegister(&PPC::G8RC_NOX0RegClass);
      BuildMI(FirstMBB, MBBI, dl, TII.get(PPC::MovePCtoLR8));
      BuildMI(FirstMBB, MBBI, dl, TII.get(PPC::MFLR8), GlobalBaseReg);
    }
  }
  return CurDAG->getRegister(GlobalBaseReg,
                             PPCLowering.getPointerTy()).getNode();
}

/// isIntS16Immediate - This method tests to see if the node is either a 32-bit
/// or 64-bit immediate, and if the value can be accurately represented as a
/// sign extension from a 16-bit value.  If so, this returns true and the
/// immediate.
static bool isIntS16Immediate(SDNode *N, short &Imm) {
  if (N->getOpcode() != ISD::Constant)
    return false;

  Imm = (short)cast<ConstantSDNode>(N)->getZExtValue();
  if (N->getValueType(0) == MVT::i32)
    return Imm == (int32_t)cast<ConstantSDNode>(N)->getZExtValue();
  else
    return Imm == (int64_t)cast<ConstantSDNode>(N)->getZExtValue();
}

static bool isIntS16Immediate(SDValue Op, short &Imm) {
  return isIntS16Immediate(Op.getNode(), Imm);
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

/// isInt64Immediate - This method tests to see if the node is a 64-bit constant
/// operand.  If so Imm will receive the 64-bit value.
static bool isInt64Immediate(SDNode *N, uint64_t &Imm) {
  if (N->getOpcode() == ISD::Constant && N->getValueType(0) == MVT::i64) {
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
  return N->getOpcode() == Opc
         && isInt32Immediate(N->getOperand(1).getNode(), Imm);
}

bool PPCDAGToDAGISel::isRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME) {
  if (!Val)
    return false;

  if (isShiftedMask_32(Val)) {
    // look for the first non-zero bit
    MB = countLeadingZeros(Val);
    // look for the first zero bit after the run of ones
    ME = countLeadingZeros((Val - 1) ^ Val);
    return true;
  } else {
    Val = ~Val; // invert mask
    if (isShiftedMask_32(Val)) {
      // effectively look for the first zero bit
      ME = countLeadingZeros(Val) - 1;
      // effectively look for the first one bit after the run of zeros
      MB = countLeadingZeros((Val - 1) ^ Val) + 1;
      return true;
    }
  }
  // no run present
  return false;
}

bool PPCDAGToDAGISel::isRotateAndMask(SDNode *N, unsigned Mask,
                                      bool isShiftMask, unsigned &SH,
                                      unsigned &MB, unsigned &ME) {
  // Don't even go down this path for i64, since different logic will be
  // necessary for rldicl/rldicr/rldimi.
  if (N->getValueType(0) != MVT::i32)
    return false;

  unsigned Shift  = 32;
  unsigned Indeterminant = ~0;  // bit mask marking indeterminant results
  unsigned Opcode = N->getOpcode();
  if (N->getNumOperands() != 2 ||
      !isInt32Immediate(N->getOperand(1).getNode(), Shift) || (Shift > 31))
    return false;

  if (Opcode == ISD::SHL) {
    // apply shift left to mask if it comes first
    if (isShiftMask) Mask = Mask << Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu << Shift);
  } else if (Opcode == ISD::SRL) {
    // apply shift right to mask if it comes first
    if (isShiftMask) Mask = Mask >> Shift;
    // determine which bits are made indeterminant by shift
    Indeterminant = ~(0xFFFFFFFFu >> Shift);
    // adjust for the left rotate
    Shift = 32 - Shift;
  } else if (Opcode == ISD::ROTL) {
    Indeterminant = 0;
  } else {
    return false;
  }

  // if the mask doesn't intersect any Indeterminant bits
  if (Mask && !(Mask & Indeterminant)) {
    SH = Shift & 31;
    // make sure the mask is still a mask (wrap arounds may not be)
    return isRunOfOnes(Mask, MB, ME);
  }
  return false;
}

/// SelectBitfieldInsert - turn an or of two masked values into
/// the rotate left word immediate then mask insert (rlwimi) instruction.
SDNode *PPCDAGToDAGISel::SelectBitfieldInsert(SDNode *N) {
  SDValue Op0 = N->getOperand(0);
  SDValue Op1 = N->getOperand(1);
  SDLoc dl(N);

  APInt LKZ, LKO, RKZ, RKO;
  CurDAG->ComputeMaskedBits(Op0, LKZ, LKO);
  CurDAG->ComputeMaskedBits(Op1, RKZ, RKO);

  unsigned TargetMask = LKZ.getZExtValue();
  unsigned InsertMask = RKZ.getZExtValue();

  if ((TargetMask | InsertMask) == 0xFFFFFFFF) {
    unsigned Op0Opc = Op0.getOpcode();
    unsigned Op1Opc = Op1.getOpcode();
    unsigned Value, SH = 0;
    TargetMask = ~TargetMask;
    InsertMask = ~InsertMask;

    // If the LHS has a foldable shift and the RHS does not, then swap it to the
    // RHS so that we can fold the shift into the insert.
    if (Op0Opc == ISD::AND && Op1Opc == ISD::AND) {
      if (Op0.getOperand(0).getOpcode() == ISD::SHL ||
          Op0.getOperand(0).getOpcode() == ISD::SRL) {
        if (Op1.getOperand(0).getOpcode() != ISD::SHL &&
            Op1.getOperand(0).getOpcode() != ISD::SRL) {
          std::swap(Op0, Op1);
          std::swap(Op0Opc, Op1Opc);
          std::swap(TargetMask, InsertMask);
        }
      }
    } else if (Op0Opc == ISD::SHL || Op0Opc == ISD::SRL) {
      if (Op1Opc == ISD::AND && Op1.getOperand(0).getOpcode() != ISD::SHL &&
          Op1.getOperand(0).getOpcode() != ISD::SRL) {
        std::swap(Op0, Op1);
        std::swap(Op0Opc, Op1Opc);
        std::swap(TargetMask, InsertMask);
      }
    }

    unsigned MB, ME;
    if (isRunOfOnes(InsertMask, MB, ME)) {
      SDValue Tmp1, Tmp2;

      if ((Op1Opc == ISD::SHL || Op1Opc == ISD::SRL) &&
          isInt32Immediate(Op1.getOperand(1), Value)) {
        Op1 = Op1.getOperand(0);
        SH  = (Op1Opc == ISD::SHL) ? Value : 32 - Value;
      }
      if (Op1Opc == ISD::AND) {
        unsigned SHOpc = Op1.getOperand(0).getOpcode();
        if ((SHOpc == ISD::SHL || SHOpc == ISD::SRL) &&
            isInt32Immediate(Op1.getOperand(0).getOperand(1), Value)) {
	  // Note that Value must be in range here (less than 32) because
	  // otherwise there would not be any bits set in InsertMask.
          Op1 = Op1.getOperand(0).getOperand(0);
          SH  = (SHOpc == ISD::SHL) ? Value : 32 - Value;
        }
      }

      SH &= 31;
      SDValue Ops[] = { Op0, Op1, getI32Imm(SH), getI32Imm(MB),
                          getI32Imm(ME) };
      return CurDAG->getMachineNode(PPC::RLWIMI, dl, MVT::i32, Ops);
    }
  }
  return 0;
}

/// SelectCC - Select a comparison of the specified values with the specified
/// condition code, returning the CR# of the expression.
SDValue PPCDAGToDAGISel::SelectCC(SDValue LHS, SDValue RHS,
                                    ISD::CondCode CC, SDLoc dl) {
  // Always select the LHS.
  unsigned Opc;

  if (LHS.getValueType() == MVT::i32) {
    unsigned Imm;
    if (CC == ISD::SETEQ || CC == ISD::SETNE) {
      if (isInt32Immediate(RHS, Imm)) {
        // SETEQ/SETNE comparison with 16-bit immediate, fold it.
        if (isUInt<16>(Imm))
          return SDValue(CurDAG->getMachineNode(PPC::CMPLWI, dl, MVT::i32, LHS,
                                                getI32Imm(Imm & 0xFFFF)), 0);
        // If this is a 16-bit signed immediate, fold it.
        if (isInt<16>((int)Imm))
          return SDValue(CurDAG->getMachineNode(PPC::CMPWI, dl, MVT::i32, LHS,
                                                getI32Imm(Imm & 0xFFFF)), 0);

        // For non-equality comparisons, the default code would materialize the
        // constant, then compare against it, like this:
        //   lis r2, 4660
        //   ori r2, r2, 22136
        //   cmpw cr0, r3, r2
        // Since we are just comparing for equality, we can emit this instead:
        //   xoris r0,r3,0x1234
        //   cmplwi cr0,r0,0x5678
        //   beq cr0,L6
        SDValue Xor(CurDAG->getMachineNode(PPC::XORIS, dl, MVT::i32, LHS,
                                           getI32Imm(Imm >> 16)), 0);
        return SDValue(CurDAG->getMachineNode(PPC::CMPLWI, dl, MVT::i32, Xor,
                                              getI32Imm(Imm & 0xFFFF)), 0);
      }
      Opc = PPC::CMPLW;
    } else if (ISD::isUnsignedIntSetCC(CC)) {
      if (isInt32Immediate(RHS, Imm) && isUInt<16>(Imm))
        return SDValue(CurDAG->getMachineNode(PPC::CMPLWI, dl, MVT::i32, LHS,
                                              getI32Imm(Imm & 0xFFFF)), 0);
      Opc = PPC::CMPLW;
    } else {
      short SImm;
      if (isIntS16Immediate(RHS, SImm))
        return SDValue(CurDAG->getMachineNode(PPC::CMPWI, dl, MVT::i32, LHS,
                                              getI32Imm((int)SImm & 0xFFFF)),
                         0);
      Opc = PPC::CMPW;
    }
  } else if (LHS.getValueType() == MVT::i64) {
    uint64_t Imm;
    if (CC == ISD::SETEQ || CC == ISD::SETNE) {
      if (isInt64Immediate(RHS.getNode(), Imm)) {
        // SETEQ/SETNE comparison with 16-bit immediate, fold it.
        if (isUInt<16>(Imm))
          return SDValue(CurDAG->getMachineNode(PPC::CMPLDI, dl, MVT::i64, LHS,
                                                getI32Imm(Imm & 0xFFFF)), 0);
        // If this is a 16-bit signed immediate, fold it.
        if (isInt<16>(Imm))
          return SDValue(CurDAG->getMachineNode(PPC::CMPDI, dl, MVT::i64, LHS,
                                                getI32Imm(Imm & 0xFFFF)), 0);

        // For non-equality comparisons, the default code would materialize the
        // constant, then compare against it, like this:
        //   lis r2, 4660
        //   ori r2, r2, 22136
        //   cmpd cr0, r3, r2
        // Since we are just comparing for equality, we can emit this instead:
        //   xoris r0,r3,0x1234
        //   cmpldi cr0,r0,0x5678
        //   beq cr0,L6
        if (isUInt<32>(Imm)) {
          SDValue Xor(CurDAG->getMachineNode(PPC::XORIS8, dl, MVT::i64, LHS,
                                             getI64Imm(Imm >> 16)), 0);
          return SDValue(CurDAG->getMachineNode(PPC::CMPLDI, dl, MVT::i64, Xor,
                                                getI64Imm(Imm & 0xFFFF)), 0);
        }
      }
      Opc = PPC::CMPLD;
    } else if (ISD::isUnsignedIntSetCC(CC)) {
      if (isInt64Immediate(RHS.getNode(), Imm) && isUInt<16>(Imm))
        return SDValue(CurDAG->getMachineNode(PPC::CMPLDI, dl, MVT::i64, LHS,
                                              getI64Imm(Imm & 0xFFFF)), 0);
      Opc = PPC::CMPLD;
    } else {
      short SImm;
      if (isIntS16Immediate(RHS, SImm))
        return SDValue(CurDAG->getMachineNode(PPC::CMPDI, dl, MVT::i64, LHS,
                                              getI64Imm(SImm & 0xFFFF)),
                         0);
      Opc = PPC::CMPD;
    }
  } else if (LHS.getValueType() == MVT::f32) {
    Opc = PPC::FCMPUS;
  } else {
    assert(LHS.getValueType() == MVT::f64 && "Unknown vt!");
    Opc = PPC::FCMPUD;
  }
  return SDValue(CurDAG->getMachineNode(Opc, dl, MVT::i32, LHS, RHS), 0);
}

static PPC::Predicate getPredicateForSetCC(ISD::CondCode CC) {
  switch (CC) {
  case ISD::SETUEQ:
  case ISD::SETONE:
  case ISD::SETOLE:
  case ISD::SETOGE:
    llvm_unreachable("Should be lowered by legalize!");
  default: llvm_unreachable("Unknown condition!");
  case ISD::SETOEQ:
  case ISD::SETEQ:  return PPC::PRED_EQ;
  case ISD::SETUNE:
  case ISD::SETNE:  return PPC::PRED_NE;
  case ISD::SETOLT:
  case ISD::SETLT:  return PPC::PRED_LT;
  case ISD::SETULE:
  case ISD::SETLE:  return PPC::PRED_LE;
  case ISD::SETOGT:
  case ISD::SETGT:  return PPC::PRED_GT;
  case ISD::SETUGE:
  case ISD::SETGE:  return PPC::PRED_GE;
  case ISD::SETO:   return PPC::PRED_NU;
  case ISD::SETUO:  return PPC::PRED_UN;
    // These two are invalid for floating point.  Assume we have int.
  case ISD::SETULT: return PPC::PRED_LT;
  case ISD::SETUGT: return PPC::PRED_GT;
  }
}

/// getCRIdxForSetCC - Return the index of the condition register field
/// associated with the SetCC condition, and whether or not the field is
/// treated as inverted.  That is, lt = 0; ge = 0 inverted.
static unsigned getCRIdxForSetCC(ISD::CondCode CC, bool &Invert) {
  Invert = false;
  switch (CC) {
  default: llvm_unreachable("Unknown condition!");
  case ISD::SETOLT:
  case ISD::SETLT:  return 0;                  // Bit #0 = SETOLT
  case ISD::SETOGT:
  case ISD::SETGT:  return 1;                  // Bit #1 = SETOGT
  case ISD::SETOEQ:
  case ISD::SETEQ:  return 2;                  // Bit #2 = SETOEQ
  case ISD::SETUO:  return 3;                  // Bit #3 = SETUO
  case ISD::SETUGE:
  case ISD::SETGE:  Invert = true; return 0;   // !Bit #0 = SETUGE
  case ISD::SETULE:
  case ISD::SETLE:  Invert = true; return 1;   // !Bit #1 = SETULE
  case ISD::SETUNE:
  case ISD::SETNE:  Invert = true; return 2;   // !Bit #2 = SETUNE
  case ISD::SETO:   Invert = true; return 3;   // !Bit #3 = SETO
  case ISD::SETUEQ:
  case ISD::SETOGE:
  case ISD::SETOLE:
  case ISD::SETONE:
    llvm_unreachable("Invalid branch code: should be expanded by legalize");
  // These are invalid for floating point.  Assume integer.
  case ISD::SETULT: return 0;
  case ISD::SETUGT: return 1;
  }
}

// getVCmpInst: return the vector compare instruction for the specified
// vector type and condition code. Since this is for altivec specific code,
// only support the altivec types (v16i8, v8i16, v4i32, and v4f32).
static unsigned int getVCmpInst(MVT::SimpleValueType VecVT, ISD::CondCode CC) {
  switch (CC) {
    case ISD::SETEQ:
    case ISD::SETUEQ:
    case ISD::SETNE:
    case ISD::SETUNE:
      if (VecVT == MVT::v16i8)
        return PPC::VCMPEQUB;
      else if (VecVT == MVT::v8i16)
        return PPC::VCMPEQUH;
      else if (VecVT == MVT::v4i32)
        return PPC::VCMPEQUW;
      // v4f32 != v4f32 could be translate to unordered not equal
      else if (VecVT == MVT::v4f32)
        return PPC::VCMPEQFP;
      break;
    case ISD::SETLT:
    case ISD::SETGT:
    case ISD::SETLE:
    case ISD::SETGE:
      if (VecVT == MVT::v16i8)
        return PPC::VCMPGTSB;
      else if (VecVT == MVT::v8i16)
        return PPC::VCMPGTSH;
      else if (VecVT == MVT::v4i32)
        return PPC::VCMPGTSW;
      else if (VecVT == MVT::v4f32)
        return PPC::VCMPGTFP;
      break;
    case ISD::SETULT:
    case ISD::SETUGT:
    case ISD::SETUGE:
    case ISD::SETULE:
      if (VecVT == MVT::v16i8)
        return PPC::VCMPGTUB;
      else if (VecVT == MVT::v8i16)
        return PPC::VCMPGTUH;
      else if (VecVT == MVT::v4i32)
        return PPC::VCMPGTUW;
      break;
    case ISD::SETOEQ:
      if (VecVT == MVT::v4f32)
        return PPC::VCMPEQFP;
      break;
    case ISD::SETOLT:
    case ISD::SETOGT:
    case ISD::SETOLE:
      if (VecVT == MVT::v4f32)
        return PPC::VCMPGTFP;
      break;
    case ISD::SETOGE:
      if (VecVT == MVT::v4f32)
        return PPC::VCMPGEFP;
      break;
    default:
      break;
  }
  llvm_unreachable("Invalid integer vector compare condition");
}

// getVCmpEQInst: return the equal compare instruction for the specified vector
// type. Since this is for altivec specific code, only support the altivec
// types (v16i8, v8i16, v4i32, and v4f32).
static unsigned int getVCmpEQInst(MVT::SimpleValueType VecVT) {
  switch (VecVT) {
    case MVT::v16i8:
      return PPC::VCMPEQUB;
    case MVT::v8i16:
      return PPC::VCMPEQUH;
    case MVT::v4i32:
      return PPC::VCMPEQUW;
    case MVT::v4f32:
      return PPC::VCMPEQFP;
    default:
      llvm_unreachable("Invalid integer vector compare condition");
  }
}


SDNode *PPCDAGToDAGISel::SelectSETCC(SDNode *N) {
  SDLoc dl(N);
  unsigned Imm;
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();
  EVT PtrVT = CurDAG->getTargetLoweringInfo().getPointerTy();
  bool isPPC64 = (PtrVT == MVT::i64);

  if (isInt32Immediate(N->getOperand(1), Imm)) {
    // We can codegen setcc op, imm very efficiently compared to a brcond.
    // Check for those cases here.
    // setcc op, 0
    if (Imm == 0) {
      SDValue Op = N->getOperand(0);
      switch (CC) {
      default: break;
      case ISD::SETEQ: {
        Op = SDValue(CurDAG->getMachineNode(PPC::CNTLZW, dl, MVT::i32, Op), 0);
        SDValue Ops[] = { Op, getI32Imm(27), getI32Imm(5), getI32Imm(31) };
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
      }
      case ISD::SETNE: {
        if (isPPC64) break;
        SDValue AD =
          SDValue(CurDAG->getMachineNode(PPC::ADDIC, dl, MVT::i32, MVT::Glue,
                                         Op, getI32Imm(~0U)), 0);
        return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, AD, Op,
                                    AD.getValue(1));
      }
      case ISD::SETLT: {
        SDValue Ops[] = { Op, getI32Imm(1), getI32Imm(31), getI32Imm(31) };
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
      }
      case ISD::SETGT: {
        SDValue T =
          SDValue(CurDAG->getMachineNode(PPC::NEG, dl, MVT::i32, Op), 0);
        T = SDValue(CurDAG->getMachineNode(PPC::ANDC, dl, MVT::i32, T, Op), 0);
        SDValue Ops[] = { T, getI32Imm(1), getI32Imm(31), getI32Imm(31) };
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
      }
      }
    } else if (Imm == ~0U) {        // setcc op, -1
      SDValue Op = N->getOperand(0);
      switch (CC) {
      default: break;
      case ISD::SETEQ:
        if (isPPC64) break;
        Op = SDValue(CurDAG->getMachineNode(PPC::ADDIC, dl, MVT::i32, MVT::Glue,
                                            Op, getI32Imm(1)), 0);
        return CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32,
                              SDValue(CurDAG->getMachineNode(PPC::LI, dl,
                                                             MVT::i32,
                                                             getI32Imm(0)), 0),
                                      Op.getValue(1));
      case ISD::SETNE: {
        if (isPPC64) break;
        Op = SDValue(CurDAG->getMachineNode(PPC::NOR, dl, MVT::i32, Op, Op), 0);
        SDNode *AD = CurDAG->getMachineNode(PPC::ADDIC, dl, MVT::i32, MVT::Glue,
                                            Op, getI32Imm(~0U));
        return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32, SDValue(AD, 0),
                                    Op, SDValue(AD, 1));
      }
      case ISD::SETLT: {
        SDValue AD = SDValue(CurDAG->getMachineNode(PPC::ADDI, dl, MVT::i32, Op,
                                                    getI32Imm(1)), 0);
        SDValue AN = SDValue(CurDAG->getMachineNode(PPC::AND, dl, MVT::i32, AD,
                                                    Op), 0);
        SDValue Ops[] = { AN, getI32Imm(1), getI32Imm(31), getI32Imm(31) };
        return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
      }
      case ISD::SETGT: {
        SDValue Ops[] = { Op, getI32Imm(1), getI32Imm(31), getI32Imm(31) };
        Op = SDValue(CurDAG->getMachineNode(PPC::RLWINM, dl, MVT::i32, Ops),
                     0);
        return CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Op,
                                    getI32Imm(1));
      }
      }
    }
  }

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // Altivec Vector compare instructions do not set any CR register by default and
  // vector compare operations return the same type as the operands.
  if (LHS.getValueType().isVector()) {
    EVT VecVT = LHS.getValueType();
    MVT::SimpleValueType VT = VecVT.getSimpleVT().SimpleTy;
    unsigned int VCmpInst = getVCmpInst(VT, CC);

    switch (CC) {
      case ISD::SETEQ:
      case ISD::SETOEQ:
      case ISD::SETUEQ:
        return CurDAG->SelectNodeTo(N, VCmpInst, VecVT, LHS, RHS);
      case ISD::SETNE:
      case ISD::SETONE:
      case ISD::SETUNE: {
        SDValue VCmp(CurDAG->getMachineNode(VCmpInst, dl, VecVT, LHS, RHS), 0);
        return CurDAG->SelectNodeTo(N, PPC::VNOR, VecVT, VCmp, VCmp);
      } 
      case ISD::SETLT:
      case ISD::SETOLT:
      case ISD::SETULT:
        return CurDAG->SelectNodeTo(N, VCmpInst, VecVT, RHS, LHS);
      case ISD::SETGT:
      case ISD::SETOGT:
      case ISD::SETUGT:
        return CurDAG->SelectNodeTo(N, VCmpInst, VecVT, LHS, RHS);
      case ISD::SETGE:
      case ISD::SETOGE:
      case ISD::SETUGE: {
        // Small optimization: Altivec provides a 'Vector Compare Greater Than
        // or Equal To' instruction (vcmpgefp), so in this case there is no
        // need for extra logic for the equal compare.
        if (VecVT.getSimpleVT().isFloatingPoint()) {
          return CurDAG->SelectNodeTo(N, VCmpInst, VecVT, LHS, RHS);
        } else {
          SDValue VCmpGT(CurDAG->getMachineNode(VCmpInst, dl, VecVT, LHS, RHS), 0);
          unsigned int VCmpEQInst = getVCmpEQInst(VT);
          SDValue VCmpEQ(CurDAG->getMachineNode(VCmpEQInst, dl, VecVT, LHS, RHS), 0);
          return CurDAG->SelectNodeTo(N, PPC::VOR, VecVT, VCmpGT, VCmpEQ);
        }
      }
      case ISD::SETLE:
      case ISD::SETOLE:
      case ISD::SETULE: {
        SDValue VCmpLE(CurDAG->getMachineNode(VCmpInst, dl, VecVT, RHS, LHS), 0);
        unsigned int VCmpEQInst = getVCmpEQInst(VT);
        SDValue VCmpEQ(CurDAG->getMachineNode(VCmpEQInst, dl, VecVT, LHS, RHS), 0);
        return CurDAG->SelectNodeTo(N, PPC::VOR, VecVT, VCmpLE, VCmpEQ);
      }
      default:
        llvm_unreachable("Invalid vector compare type: should be expanded by legalize");
    }
  }

  bool Inv;
  unsigned Idx = getCRIdxForSetCC(CC, Inv);
  SDValue CCReg = SelectCC(LHS, RHS, CC, dl);
  SDValue IntCR;

  // Force the ccreg into CR7.
  SDValue CR7Reg = CurDAG->getRegister(PPC::CR7, MVT::i32);

  SDValue InFlag(0, 0);  // Null incoming flag value.
  CCReg = CurDAG->getCopyToReg(CurDAG->getEntryNode(), dl, CR7Reg, CCReg,
                               InFlag).getValue(1);

  IntCR = SDValue(CurDAG->getMachineNode(PPC::MFOCRF, dl, MVT::i32, CR7Reg,
                                         CCReg), 0);

  SDValue Ops[] = { IntCR, getI32Imm((32-(3-Idx)) & 31),
                      getI32Imm(31), getI32Imm(31) };
  if (!Inv)
    return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);

  // Get the specified bit.
  SDValue Tmp =
    SDValue(CurDAG->getMachineNode(PPC::RLWINM, dl, MVT::i32, Ops), 0);
  return CurDAG->SelectNodeTo(N, PPC::XORI, MVT::i32, Tmp, getI32Imm(1));
}


// Select - Convert the specified operand from a target-independent to a
// target-specific node if it hasn't already been changed.
SDNode *PPCDAGToDAGISel::Select(SDNode *N) {
  SDLoc dl(N);
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return NULL;   // Already selected.
  }

  switch (N->getOpcode()) {
  default: break;

  case ISD::Constant: {
    if (N->getValueType(0) == MVT::i64) {
      // Get 64 bit value.
      int64_t Imm = cast<ConstantSDNode>(N)->getZExtValue();
      // Assume no remaining bits.
      unsigned Remainder = 0;
      // Assume no shift required.
      unsigned Shift = 0;

      // If it can't be represented as a 32 bit value.
      if (!isInt<32>(Imm)) {
        Shift = countTrailingZeros<uint64_t>(Imm);
        int64_t ImmSh = static_cast<uint64_t>(Imm) >> Shift;

        // If the shifted value fits 32 bits.
        if (isInt<32>(ImmSh)) {
          // Go with the shifted value.
          Imm = ImmSh;
        } else {
          // Still stuck with a 64 bit value.
          Remainder = Imm;
          Shift = 32;
          Imm >>= 32;
        }
      }

      // Intermediate operand.
      SDNode *Result;

      // Handle first 32 bits.
      unsigned Lo = Imm & 0xFFFF;
      unsigned Hi = (Imm >> 16) & 0xFFFF;

      // Simple value.
      if (isInt<16>(Imm)) {
       // Just the Lo bits.
        Result = CurDAG->getMachineNode(PPC::LI8, dl, MVT::i64, getI32Imm(Lo));
      } else if (Lo) {
        // Handle the Hi bits.
        unsigned OpC = Hi ? PPC::LIS8 : PPC::LI8;
        Result = CurDAG->getMachineNode(OpC, dl, MVT::i64, getI32Imm(Hi));
        // And Lo bits.
        Result = CurDAG->getMachineNode(PPC::ORI8, dl, MVT::i64,
                                        SDValue(Result, 0), getI32Imm(Lo));
      } else {
       // Just the Hi bits.
        Result = CurDAG->getMachineNode(PPC::LIS8, dl, MVT::i64, getI32Imm(Hi));
      }

      // If no shift, we're done.
      if (!Shift) return Result;

      // Shift for next step if the upper 32-bits were not zero.
      if (Imm) {
        Result = CurDAG->getMachineNode(PPC::RLDICR, dl, MVT::i64,
                                        SDValue(Result, 0),
                                        getI32Imm(Shift),
                                        getI32Imm(63 - Shift));
      }

      // Add in the last bits as required.
      if ((Hi = (Remainder >> 16) & 0xFFFF)) {
        Result = CurDAG->getMachineNode(PPC::ORIS8, dl, MVT::i64,
                                        SDValue(Result, 0), getI32Imm(Hi));
      }
      if ((Lo = Remainder & 0xFFFF)) {
        Result = CurDAG->getMachineNode(PPC::ORI8, dl, MVT::i64,
                                        SDValue(Result, 0), getI32Imm(Lo));
      }

      return Result;
    }
    break;
  }

  case ISD::SETCC:
    return SelectSETCC(N);
  case PPCISD::GlobalBaseReg:
    return getGlobalBaseReg();

  case ISD::FrameIndex: {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, N->getValueType(0));
    unsigned Opc = N->getValueType(0) == MVT::i32 ? PPC::ADDI : PPC::ADDI8;
    if (N->hasOneUse())
      return CurDAG->SelectNodeTo(N, Opc, N->getValueType(0), TFI,
                                  getSmallIPtrImm(0));
    return CurDAG->getMachineNode(Opc, dl, N->getValueType(0), TFI,
                                  getSmallIPtrImm(0));
  }

  case PPCISD::MFOCRF: {
    SDValue InFlag = N->getOperand(1);
    return CurDAG->getMachineNode(PPC::MFOCRF, dl, MVT::i32,
                                  N->getOperand(0), InFlag);
  }

  case ISD::SDIV: {
    // FIXME: since this depends on the setting of the carry flag from the srawi
    //        we should really be making notes about that for the scheduler.
    // FIXME: It sure would be nice if we could cheaply recognize the
    //        srl/add/sra pattern the dag combiner will generate for this as
    //        sra/addze rather than having to handle sdiv ourselves.  oh well.
    unsigned Imm;
    if (isInt32Immediate(N->getOperand(1), Imm)) {
      SDValue N0 = N->getOperand(0);
      if ((signed)Imm > 0 && isPowerOf2_32(Imm)) {
        SDNode *Op =
          CurDAG->getMachineNode(PPC::SRAWI, dl, MVT::i32, MVT::Glue,
                                 N0, getI32Imm(Log2_32(Imm)));
        return CurDAG->SelectNodeTo(N, PPC::ADDZE, MVT::i32,
                                    SDValue(Op, 0), SDValue(Op, 1));
      } else if ((signed)Imm < 0 && isPowerOf2_32(-Imm)) {
        SDNode *Op =
          CurDAG->getMachineNode(PPC::SRAWI, dl, MVT::i32, MVT::Glue,
                                 N0, getI32Imm(Log2_32(-Imm)));
        SDValue PT =
          SDValue(CurDAG->getMachineNode(PPC::ADDZE, dl, MVT::i32,
                                         SDValue(Op, 0), SDValue(Op, 1)),
                    0);
        return CurDAG->SelectNodeTo(N, PPC::NEG, MVT::i32, PT);
      }
    }

    // Other cases are autogenerated.
    break;
  }

  case ISD::LOAD: {
    // Handle preincrement loads.
    LoadSDNode *LD = cast<LoadSDNode>(N);
    EVT LoadedVT = LD->getMemoryVT();

    // Normal loads are handled by code generated from the .td file.
    if (LD->getAddressingMode() != ISD::PRE_INC)
      break;

    SDValue Offset = LD->getOffset();
    if (Offset.getOpcode() == ISD::TargetConstant ||
        Offset.getOpcode() == ISD::TargetGlobalAddress) {

      unsigned Opcode;
      bool isSExt = LD->getExtensionType() == ISD::SEXTLOAD;
      if (LD->getValueType(0) != MVT::i64) {
        // Handle PPC32 integer and normal FP loads.
        assert((!isSExt || LoadedVT == MVT::i16) && "Invalid sext update load");
        switch (LoadedVT.getSimpleVT().SimpleTy) {
          default: llvm_unreachable("Invalid PPC load type!");
          case MVT::f64: Opcode = PPC::LFDU; break;
          case MVT::f32: Opcode = PPC::LFSU; break;
          case MVT::i32: Opcode = PPC::LWZU; break;
          case MVT::i16: Opcode = isSExt ? PPC::LHAU : PPC::LHZU; break;
          case MVT::i1:
          case MVT::i8:  Opcode = PPC::LBZU; break;
        }
      } else {
        assert(LD->getValueType(0) == MVT::i64 && "Unknown load result type!");
        assert((!isSExt || LoadedVT == MVT::i16) && "Invalid sext update load");
        switch (LoadedVT.getSimpleVT().SimpleTy) {
          default: llvm_unreachable("Invalid PPC load type!");
          case MVT::i64: Opcode = PPC::LDU; break;
          case MVT::i32: Opcode = PPC::LWZU8; break;
          case MVT::i16: Opcode = isSExt ? PPC::LHAU8 : PPC::LHZU8; break;
          case MVT::i1:
          case MVT::i8:  Opcode = PPC::LBZU8; break;
        }
      }

      SDValue Chain = LD->getChain();
      SDValue Base = LD->getBasePtr();
      SDValue Ops[] = { Offset, Base, Chain };
      return CurDAG->getMachineNode(Opcode, dl, LD->getValueType(0),
                                    PPCLowering.getPointerTy(),
                                    MVT::Other, Ops);
    } else {
      unsigned Opcode;
      bool isSExt = LD->getExtensionType() == ISD::SEXTLOAD;
      if (LD->getValueType(0) != MVT::i64) {
        // Handle PPC32 integer and normal FP loads.
        assert((!isSExt || LoadedVT == MVT::i16) && "Invalid sext update load");
        switch (LoadedVT.getSimpleVT().SimpleTy) {
          default: llvm_unreachable("Invalid PPC load type!");
          case MVT::f64: Opcode = PPC::LFDUX; break;
          case MVT::f32: Opcode = PPC::LFSUX; break;
          case MVT::i32: Opcode = PPC::LWZUX; break;
          case MVT::i16: Opcode = isSExt ? PPC::LHAUX : PPC::LHZUX; break;
          case MVT::i1:
          case MVT::i8:  Opcode = PPC::LBZUX; break;
        }
      } else {
        assert(LD->getValueType(0) == MVT::i64 && "Unknown load result type!");
        assert((!isSExt || LoadedVT == MVT::i16 || LoadedVT == MVT::i32) &&
               "Invalid sext update load");
        switch (LoadedVT.getSimpleVT().SimpleTy) {
          default: llvm_unreachable("Invalid PPC load type!");
          case MVT::i64: Opcode = PPC::LDUX; break;
          case MVT::i32: Opcode = isSExt ? PPC::LWAUX  : PPC::LWZUX8; break;
          case MVT::i16: Opcode = isSExt ? PPC::LHAUX8 : PPC::LHZUX8; break;
          case MVT::i1:
          case MVT::i8:  Opcode = PPC::LBZUX8; break;
        }
      }

      SDValue Chain = LD->getChain();
      SDValue Base = LD->getBasePtr();
      SDValue Ops[] = { Base, Offset, Chain };
      return CurDAG->getMachineNode(Opcode, dl, LD->getValueType(0),
                                    PPCLowering.getPointerTy(),
                                    MVT::Other, Ops);
    }
  }

  case ISD::AND: {
    unsigned Imm, Imm2, SH, MB, ME;
    uint64_t Imm64;

    // If this is an and of a value rotated between 0 and 31 bits and then and'd
    // with a mask, emit rlwinm
    if (isInt32Immediate(N->getOperand(1), Imm) &&
        isRotateAndMask(N->getOperand(0).getNode(), Imm, false, SH, MB, ME)) {
      SDValue Val = N->getOperand(0).getOperand(0);
      SDValue Ops[] = { Val, getI32Imm(SH), getI32Imm(MB), getI32Imm(ME) };
      return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
    }
    // If this is just a masked value where the input is not handled above, and
    // is not a rotate-left (handled by a pattern in the .td file), emit rlwinm
    if (isInt32Immediate(N->getOperand(1), Imm) &&
        isRunOfOnes(Imm, MB, ME) &&
        N->getOperand(0).getOpcode() != ISD::ROTL) {
      SDValue Val = N->getOperand(0);
      SDValue Ops[] = { Val, getI32Imm(0), getI32Imm(MB), getI32Imm(ME) };
      return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
    }
    // If this is a 64-bit zero-extension mask, emit rldicl.
    if (isInt64Immediate(N->getOperand(1).getNode(), Imm64) &&
        isMask_64(Imm64)) {
      SDValue Val = N->getOperand(0);
      MB = 64 - CountTrailingOnes_64(Imm64);
      SDValue Ops[] = { Val, getI32Imm(0), getI32Imm(MB) };
      return CurDAG->SelectNodeTo(N, PPC::RLDICL, MVT::i64, Ops, 3);
    }
    // AND X, 0 -> 0, not "rlwinm 32".
    if (isInt32Immediate(N->getOperand(1), Imm) && (Imm == 0)) {
      ReplaceUses(SDValue(N, 0), N->getOperand(1));
      return NULL;
    }
    // ISD::OR doesn't get all the bitfield insertion fun.
    // (and (or x, c1), c2) where isRunOfOnes(~(c1^c2)) is a bitfield insert
    if (isInt32Immediate(N->getOperand(1), Imm) &&
        N->getOperand(0).getOpcode() == ISD::OR &&
        isInt32Immediate(N->getOperand(0).getOperand(1), Imm2)) {
      unsigned MB, ME;
      Imm = ~(Imm^Imm2);
      if (isRunOfOnes(Imm, MB, ME)) {
        SDValue Ops[] = { N->getOperand(0).getOperand(0),
                            N->getOperand(0).getOperand(1),
                            getI32Imm(0), getI32Imm(MB),getI32Imm(ME) };
        return CurDAG->getMachineNode(PPC::RLWIMI, dl, MVT::i32, Ops);
      }
    }

    // Other cases are autogenerated.
    break;
  }
  case ISD::OR:
    if (N->getValueType(0) == MVT::i32)
      if (SDNode *I = SelectBitfieldInsert(N))
        return I;

    // Other cases are autogenerated.
    break;
  case ISD::SHL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME)) {
      SDValue Ops[] = { N->getOperand(0).getOperand(0),
                          getI32Imm(SH), getI32Imm(MB), getI32Imm(ME) };
      return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
    }

    // Other cases are autogenerated.
    break;
  }
  case ISD::SRL: {
    unsigned Imm, SH, MB, ME;
    if (isOpcWithIntImmediate(N->getOperand(0).getNode(), ISD::AND, Imm) &&
        isRotateAndMask(N, Imm, true, SH, MB, ME)) {
      SDValue Ops[] = { N->getOperand(0).getOperand(0),
                          getI32Imm(SH), getI32Imm(MB), getI32Imm(ME) };
      return CurDAG->SelectNodeTo(N, PPC::RLWINM, MVT::i32, Ops, 4);
    }

    // Other cases are autogenerated.
    break;
  }
  case ISD::SELECT_CC: {
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(4))->get();
    EVT PtrVT = CurDAG->getTargetLoweringInfo().getPointerTy();
    bool isPPC64 = (PtrVT == MVT::i64);

    // Handle the setcc cases here.  select_cc lhs, 0, 1, 0, cc
    if (!isPPC64)
      if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N->getOperand(1)))
        if (ConstantSDNode *N2C = dyn_cast<ConstantSDNode>(N->getOperand(2)))
          if (ConstantSDNode *N3C = dyn_cast<ConstantSDNode>(N->getOperand(3)))
            if (N1C->isNullValue() && N3C->isNullValue() &&
                N2C->getZExtValue() == 1ULL && CC == ISD::SETNE &&
                // FIXME: Implement this optzn for PPC64.
                N->getValueType(0) == MVT::i32) {
              SDNode *Tmp =
                CurDAG->getMachineNode(PPC::ADDIC, dl, MVT::i32, MVT::Glue,
                                       N->getOperand(0), getI32Imm(~0U));
              return CurDAG->SelectNodeTo(N, PPC::SUBFE, MVT::i32,
                                          SDValue(Tmp, 0), N->getOperand(0),
                                          SDValue(Tmp, 1));
            }

    SDValue CCReg = SelectCC(N->getOperand(0), N->getOperand(1), CC, dl);
    unsigned BROpc = getPredicateForSetCC(CC);

    unsigned SelectCCOp;
    if (N->getValueType(0) == MVT::i32)
      SelectCCOp = PPC::SELECT_CC_I4;
    else if (N->getValueType(0) == MVT::i64)
      SelectCCOp = PPC::SELECT_CC_I8;
    else if (N->getValueType(0) == MVT::f32)
      SelectCCOp = PPC::SELECT_CC_F4;
    else if (N->getValueType(0) == MVT::f64)
      SelectCCOp = PPC::SELECT_CC_F8;
    else
      SelectCCOp = PPC::SELECT_CC_VRRC;

    SDValue Ops[] = { CCReg, N->getOperand(2), N->getOperand(3),
                        getI32Imm(BROpc) };
    return CurDAG->SelectNodeTo(N, SelectCCOp, N->getValueType(0), Ops, 4);
  }
  case PPCISD::BDNZ:
  case PPCISD::BDZ: {
    bool IsPPC64 = PPCSubTarget.isPPC64();
    SDValue Ops[] = { N->getOperand(1), N->getOperand(0) };
    return CurDAG->SelectNodeTo(N, N->getOpcode() == PPCISD::BDNZ ?
                                   (IsPPC64 ? PPC::BDNZ8 : PPC::BDNZ) :
                                   (IsPPC64 ? PPC::BDZ8 : PPC::BDZ),
                                MVT::Other, Ops, 2);
  }
  case PPCISD::COND_BRANCH: {
    // Op #0 is the Chain.
    // Op #1 is the PPC::PRED_* number.
    // Op #2 is the CR#
    // Op #3 is the Dest MBB
    // Op #4 is the Flag.
    // Prevent PPC::PRED_* from being selected into LI.
    SDValue Pred =
      getI32Imm(cast<ConstantSDNode>(N->getOperand(1))->getZExtValue());
    SDValue Ops[] = { Pred, N->getOperand(2), N->getOperand(3),
      N->getOperand(0), N->getOperand(4) };
    return CurDAG->SelectNodeTo(N, PPC::BCC, MVT::Other, Ops, 5);
  }
  case ISD::BR_CC: {
    ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(1))->get();
    SDValue CondCode = SelectCC(N->getOperand(2), N->getOperand(3), CC, dl);
    SDValue Ops[] = { getI32Imm(getPredicateForSetCC(CC)), CondCode,
                        N->getOperand(4), N->getOperand(0) };
    return CurDAG->SelectNodeTo(N, PPC::BCC, MVT::Other, Ops, 4);
  }
  case ISD::BRIND: {
    // FIXME: Should custom lower this.
    SDValue Chain = N->getOperand(0);
    SDValue Target = N->getOperand(1);
    unsigned Opc = Target.getValueType() == MVT::i32 ? PPC::MTCTR : PPC::MTCTR8;
    unsigned Reg = Target.getValueType() == MVT::i32 ? PPC::BCTR : PPC::BCTR8;
    Chain = SDValue(CurDAG->getMachineNode(Opc, dl, MVT::Glue, Target,
                                           Chain), 0);
    return CurDAG->SelectNodeTo(N, Reg, MVT::Other, Chain);
  }
  case PPCISD::TOC_ENTRY: {
    assert (PPCSubTarget.isPPC64() && "Only supported for 64-bit ABI");

    // For medium and large code model, we generate two instructions as
    // described below.  Otherwise we allow SelectCodeCommon to handle this,
    // selecting one of LDtoc, LDtocJTI, and LDtocCPT.
    CodeModel::Model CModel = TM.getCodeModel();
    if (CModel != CodeModel::Medium && CModel != CodeModel::Large)
      break;

    // The first source operand is a TargetGlobalAddress or a
    // TargetJumpTable.  If it is an externally defined symbol, a symbol
    // with common linkage, a function address, or a jump table address,
    // or if we are generating code for large code model, we generate:
    //   LDtocL(<ga:@sym>, ADDIStocHA(%X2, <ga:@sym>))
    // Otherwise we generate:
    //   ADDItocL(ADDIStocHA(%X2, <ga:@sym>), <ga:@sym>)
    SDValue GA = N->getOperand(0);
    SDValue TOCbase = N->getOperand(1);
    SDNode *Tmp = CurDAG->getMachineNode(PPC::ADDIStocHA, dl, MVT::i64,
                                        TOCbase, GA);

    if (isa<JumpTableSDNode>(GA) || CModel == CodeModel::Large)
      return CurDAG->getMachineNode(PPC::LDtocL, dl, MVT::i64, GA,
                                    SDValue(Tmp, 0));

    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(GA)) {
      const GlobalValue *GValue = G->getGlobal();
      const GlobalAlias *GAlias = dyn_cast<GlobalAlias>(GValue);
      const GlobalValue *RealGValue = GAlias ?
        GAlias->resolveAliasedGlobal(false) : GValue;
      const GlobalVariable *GVar = dyn_cast<GlobalVariable>(RealGValue);
      assert((GVar || isa<Function>(RealGValue)) &&
             "Unexpected global value subclass!");

      // An external variable is one without an initializer.  For these,
      // for variables with common linkage, and for Functions, generate
      // the LDtocL form.
      if (!GVar || !GVar->hasInitializer() || RealGValue->hasCommonLinkage() ||
          RealGValue->hasAvailableExternallyLinkage())
        return CurDAG->getMachineNode(PPC::LDtocL, dl, MVT::i64, GA,
                                      SDValue(Tmp, 0));
    }

    return CurDAG->getMachineNode(PPC::ADDItocL, dl, MVT::i64,
                                  SDValue(Tmp, 0), GA);
  }
  case PPCISD::VADD_SPLAT: {
    // This expands into one of three sequences, depending on whether
    // the first operand is odd or even, positive or negative.
    assert(isa<ConstantSDNode>(N->getOperand(0)) &&
           isa<ConstantSDNode>(N->getOperand(1)) &&
           "Invalid operand on VADD_SPLAT!");

    int Elt     = N->getConstantOperandVal(0);
    int EltSize = N->getConstantOperandVal(1);
    unsigned Opc1, Opc2, Opc3;
    EVT VT;

    if (EltSize == 1) {
      Opc1 = PPC::VSPLTISB;
      Opc2 = PPC::VADDUBM;
      Opc3 = PPC::VSUBUBM;
      VT = MVT::v16i8;
    } else if (EltSize == 2) {
      Opc1 = PPC::VSPLTISH;
      Opc2 = PPC::VADDUHM;
      Opc3 = PPC::VSUBUHM;
      VT = MVT::v8i16;
    } else {
      assert(EltSize == 4 && "Invalid element size on VADD_SPLAT!");
      Opc1 = PPC::VSPLTISW;
      Opc2 = PPC::VADDUWM;
      Opc3 = PPC::VSUBUWM;
      VT = MVT::v4i32;
    }

    if ((Elt & 1) == 0) {
      // Elt is even, in the range [-32,-18] + [16,30].
      //
      // Convert: VADD_SPLAT elt, size
      // Into:    tmp = VSPLTIS[BHW] elt
      //          VADDU[BHW]M tmp, tmp
      // Where:   [BHW] = B for size = 1, H for size = 2, W for size = 4
      SDValue EltVal = getI32Imm(Elt >> 1);
      SDNode *Tmp = CurDAG->getMachineNode(Opc1, dl, VT, EltVal);
      SDValue TmpVal = SDValue(Tmp, 0);
      return CurDAG->getMachineNode(Opc2, dl, VT, TmpVal, TmpVal);

    } else if (Elt > 0) {
      // Elt is odd and positive, in the range [17,31].
      //
      // Convert: VADD_SPLAT elt, size
      // Into:    tmp1 = VSPLTIS[BHW] elt-16
      //          tmp2 = VSPLTIS[BHW] -16
      //          VSUBU[BHW]M tmp1, tmp2
      SDValue EltVal = getI32Imm(Elt - 16);
      SDNode *Tmp1 = CurDAG->getMachineNode(Opc1, dl, VT, EltVal);
      EltVal = getI32Imm(-16);
      SDNode *Tmp2 = CurDAG->getMachineNode(Opc1, dl, VT, EltVal);
      return CurDAG->getMachineNode(Opc3, dl, VT, SDValue(Tmp1, 0),
                                    SDValue(Tmp2, 0));

    } else {
      // Elt is odd and negative, in the range [-31,-17].
      //
      // Convert: VADD_SPLAT elt, size
      // Into:    tmp1 = VSPLTIS[BHW] elt+16
      //          tmp2 = VSPLTIS[BHW] -16
      //          VADDU[BHW]M tmp1, tmp2
      SDValue EltVal = getI32Imm(Elt + 16);
      SDNode *Tmp1 = CurDAG->getMachineNode(Opc1, dl, VT, EltVal);
      EltVal = getI32Imm(-16);
      SDNode *Tmp2 = CurDAG->getMachineNode(Opc1, dl, VT, EltVal);
      return CurDAG->getMachineNode(Opc2, dl, VT, SDValue(Tmp1, 0),
                                    SDValue(Tmp2, 0));
    }
  }
  }

  return SelectCode(N);
}

/// PostProcessISelDAG - Perform some late peephole optimizations
/// on the DAG representation.
void PPCDAGToDAGISel::PostprocessISelDAG() {

  // Skip peepholes at -O0.
  if (TM.getOptLevel() == CodeGenOpt::None)
    return;

  // These optimizations are currently supported only for 64-bit SVR4.
  if (PPCSubTarget.isDarwin() || !PPCSubTarget.isPPC64())
    return;

  SelectionDAG::allnodes_iterator Position(CurDAG->getRoot().getNode());
  ++Position;

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = --Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    unsigned FirstOp;
    unsigned StorageOpcode = N->getMachineOpcode();

    switch (StorageOpcode) {
    default: continue;

    case PPC::LBZ:
    case PPC::LBZ8:
    case PPC::LD:
    case PPC::LFD:
    case PPC::LFS:
    case PPC::LHA:
    case PPC::LHA8:
    case PPC::LHZ:
    case PPC::LHZ8:
    case PPC::LWA:
    case PPC::LWZ:
    case PPC::LWZ8:
      FirstOp = 0;
      break;

    case PPC::STB:
    case PPC::STB8:
    case PPC::STD:
    case PPC::STFD:
    case PPC::STFS:
    case PPC::STH:
    case PPC::STH8:
    case PPC::STW:
    case PPC::STW8:
      FirstOp = 1;
      break;
    }

    // If this is a load or store with a zero offset, we may be able to
    // fold an add-immediate into the memory operation.
    if (!isa<ConstantSDNode>(N->getOperand(FirstOp)) ||
        N->getConstantOperandVal(FirstOp) != 0)
      continue;

    SDValue Base = N->getOperand(FirstOp + 1);
    if (!Base.isMachineOpcode())
      continue;

    unsigned Flags = 0;
    bool ReplaceFlags = true;

    // When the feeding operation is an add-immediate of some sort,
    // determine whether we need to add relocation information to the
    // target flags on the immediate operand when we fold it into the
    // load instruction.
    //
    // For something like ADDItocL, the relocation information is
    // inferred from the opcode; when we process it in the AsmPrinter,
    // we add the necessary relocation there.  A load, though, can receive
    // relocation from various flavors of ADDIxxx, so we need to carry
    // the relocation information in the target flags.
    switch (Base.getMachineOpcode()) {
    default: continue;

    case PPC::ADDI8:
    case PPC::ADDI:
      // In some cases (such as TLS) the relocation information
      // is already in place on the operand, so copying the operand
      // is sufficient.
      ReplaceFlags = false;
      // For these cases, the immediate may not be divisible by 4, in
      // which case the fold is illegal for DS-form instructions.  (The
      // other cases provide aligned addresses and are always safe.)
      if ((StorageOpcode == PPC::LWA ||
           StorageOpcode == PPC::LD  ||
           StorageOpcode == PPC::STD) &&
          (!isa<ConstantSDNode>(Base.getOperand(1)) ||
           Base.getConstantOperandVal(1) % 4 != 0))
        continue;
      break;
    case PPC::ADDIdtprelL:
      Flags = PPCII::MO_DTPREL_LO;
      break;
    case PPC::ADDItlsldL:
      Flags = PPCII::MO_TLSLD_LO;
      break;
    case PPC::ADDItocL:
      Flags = PPCII::MO_TOC_LO;
      break;
    }

    // We found an opportunity.  Reverse the operands from the add
    // immediate and substitute them into the load or store.  If
    // needed, update the target flags for the immediate operand to
    // reflect the necessary relocation information.
    DEBUG(dbgs() << "Folding add-immediate into mem-op:\nBase:    ");
    DEBUG(Base->dump(CurDAG));
    DEBUG(dbgs() << "\nN: ");
    DEBUG(N->dump(CurDAG));
    DEBUG(dbgs() << "\n");

    SDValue ImmOpnd = Base.getOperand(1);

    // If the relocation information isn't already present on the
    // immediate operand, add it now.
    if (ReplaceFlags) {
      if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(ImmOpnd)) {
        SDLoc dl(GA);
        const GlobalValue *GV = GA->getGlobal();
        // We can't perform this optimization for data whose alignment
        // is insufficient for the instruction encoding.
        if (GV->getAlignment() < 4 &&
            (StorageOpcode == PPC::LD || StorageOpcode == PPC::STD ||
             StorageOpcode == PPC::LWA)) {
          DEBUG(dbgs() << "Rejected this candidate for alignment.\n\n");
          continue;
        }
        ImmOpnd = CurDAG->getTargetGlobalAddress(GV, dl, MVT::i64, 0, Flags);
      } else if (ConstantPoolSDNode *CP =
                 dyn_cast<ConstantPoolSDNode>(ImmOpnd)) {
        const Constant *C = CP->getConstVal();
        ImmOpnd = CurDAG->getTargetConstantPool(C, MVT::i64,
                                                CP->getAlignment(),
                                                0, Flags);
      }
    }

    if (FirstOp == 1) // Store
      (void)CurDAG->UpdateNodeOperands(N, N->getOperand(0), ImmOpnd,
                                       Base.getOperand(0), N->getOperand(3));
    else // Load
      (void)CurDAG->UpdateNodeOperands(N, ImmOpnd, Base.getOperand(0),
                                       N->getOperand(2));

    // The add-immediate may now be dead, in which case remove it.
    if (Base.getNode()->use_empty())
      CurDAG->RemoveDeadNode(Base.getNode());
  }
}


/// createPPCISelDag - This pass converts a legalized DAG into a
/// PowerPC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createPPCISelDag(PPCTargetMachine &TM) {
  return new PPCDAGToDAGISel(TM);
}

static void initializePassOnce(PassRegistry &Registry) {
  const char *Name = "PowerPC DAG->DAG Pattern Instruction Selection";
  PassInfo *PI = new PassInfo(Name, "ppc-codegen", &SelectionDAGISel::ID, 0,
                              false, false);
  Registry.registerPass(*PI, true);
}

void llvm::initializePPCDAGToDAGISelPass(PassRegistry &Registry) {
  CALL_ONCE_INITIALIZATION(initializePassOnce);
}

