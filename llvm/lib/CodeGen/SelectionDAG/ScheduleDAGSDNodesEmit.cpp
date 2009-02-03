//===---- ScheduleDAGEmit.cpp - Emit routines for the ScheduleDAG class ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the Emit routines for the ScheduleDAG class, which creates
// MachineInstrs according to the computed schedule.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAGSDNodes.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// getInstrOperandRegClass - Return register class of the operand of an
/// instruction of the specified TargetInstrDesc.
static const TargetRegisterClass*
getInstrOperandRegClass(const TargetRegisterInfo *TRI, 
                        const TargetInstrInfo *TII, const TargetInstrDesc &II,
                        unsigned Op) {
  if (Op >= II.getNumOperands()) {
    assert(II.isVariadic() && "Invalid operand # of instruction");
    return NULL;
  }
  if (II.OpInfo[Op].isLookupPtrRegClass())
    return TII->getPointerRegClass();
  return TRI->getRegClass(II.OpInfo[Op].RegClass);
}

/// EmitCopyFromReg - Generate machine code for an CopyFromReg node or an
/// implicit physical register output.
void ScheduleDAGSDNodes::EmitCopyFromReg(SDNode *Node, unsigned ResNo,
                                         bool IsClone, bool IsCloned,
                                         unsigned SrcReg,
                                         DenseMap<SDValue, unsigned> &VRBaseMap) {
  unsigned VRBase = 0;
  if (TargetRegisterInfo::isVirtualRegister(SrcReg)) {
    // Just use the input register directly!
    SDValue Op(Node, ResNo);
    if (IsClone)
      VRBaseMap.erase(Op);
    bool isNew = VRBaseMap.insert(std::make_pair(Op, SrcReg)).second;
    isNew = isNew; // Silence compiler warning.
    assert(isNew && "Node emitted out of order - early");
    return;
  }

  // If the node is only used by a CopyToReg and the dest reg is a vreg, use
  // the CopyToReg'd destination register instead of creating a new vreg.
  bool MatchReg = true;
  const TargetRegisterClass *UseRC = NULL;
  if (!IsClone && !IsCloned)
    for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
         UI != E; ++UI) {
      SDNode *User = *UI;
      bool Match = true;
      if (User->getOpcode() == ISD::CopyToReg && 
          User->getOperand(2).getNode() == Node &&
          User->getOperand(2).getResNo() == ResNo) {
        unsigned DestReg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
        if (TargetRegisterInfo::isVirtualRegister(DestReg)) {
          VRBase = DestReg;
          Match = false;
        } else if (DestReg != SrcReg)
          Match = false;
      } else {
        for (unsigned i = 0, e = User->getNumOperands(); i != e; ++i) {
          SDValue Op = User->getOperand(i);
          if (Op.getNode() != Node || Op.getResNo() != ResNo)
            continue;
          MVT VT = Node->getValueType(Op.getResNo());
          if (VT == MVT::Other || VT == MVT::Flag)
            continue;
          Match = false;
          if (User->isMachineOpcode()) {
            const TargetInstrDesc &II = TII->get(User->getMachineOpcode());
            const TargetRegisterClass *RC =
              getInstrOperandRegClass(TRI,TII,II,i+II.getNumDefs());
            if (!UseRC)
              UseRC = RC;
            else if (RC)
              assert(UseRC == RC &&
                     "Multiple uses expecting different register classes!");
          }
        }
      }
      MatchReg &= Match;
      if (VRBase)
        break;
    }

  MVT VT = Node->getValueType(ResNo);
  const TargetRegisterClass *SrcRC = 0, *DstRC = 0;
  SrcRC = TRI->getPhysicalRegisterRegClass(SrcReg, VT);
  
  // Figure out the register class to create for the destreg.
  if (VRBase) {
    DstRC = MRI.getRegClass(VRBase);
  } else if (UseRC) {
    assert(UseRC->hasType(VT) && "Incompatible phys register def and uses!");
    DstRC = UseRC;
  } else {
    DstRC = TLI->getRegClassFor(VT);
  }
    
  // If all uses are reading from the src physical register and copying the
  // register is either impossible or very expensive, then don't create a copy.
  if (MatchReg && SrcRC->getCopyCost() < 0) {
    VRBase = SrcReg;
  } else {
    // Create the reg, emit the copy.
    VRBase = MRI.createVirtualRegister(DstRC);
    bool Emitted =
      TII->copyRegToReg(*BB, End, VRBase, SrcReg, DstRC, SrcRC);
    Emitted = Emitted; // Silence compiler warning.
    assert(Emitted && "Unable to issue a copy instruction!");
  }

  SDValue Op(Node, ResNo);
  if (IsClone)
    VRBaseMap.erase(Op);
  bool isNew = VRBaseMap.insert(std::make_pair(Op, VRBase)).second;
  isNew = isNew; // Silence compiler warning.
  assert(isNew && "Node emitted out of order - early");
}

/// getDstOfCopyToRegUse - If the only use of the specified result number of
/// node is a CopyToReg, return its destination register. Return 0 otherwise.
unsigned ScheduleDAGSDNodes::getDstOfOnlyCopyToRegUse(SDNode *Node,
                                                      unsigned ResNo) const {
  if (!Node->hasOneUse())
    return 0;

  SDNode *User = *Node->use_begin();
  if (User->getOpcode() == ISD::CopyToReg && 
      User->getOperand(2).getNode() == Node &&
      User->getOperand(2).getResNo() == ResNo) {
    unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      return Reg;
  }
  return 0;
}

void ScheduleDAGSDNodes::CreateVirtualRegisters(SDNode *Node, MachineInstr *MI,
                                       const TargetInstrDesc &II,
                                       bool IsClone, bool IsCloned,
                                       DenseMap<SDValue, unsigned> &VRBaseMap) {
  assert(Node->getMachineOpcode() != TargetInstrInfo::IMPLICIT_DEF &&
         "IMPLICIT_DEF should have been handled as a special case elsewhere!");

  for (unsigned i = 0; i < II.getNumDefs(); ++i) {
    // If the specific node value is only used by a CopyToReg and the dest reg
    // is a vreg, use the CopyToReg'd destination register instead of creating
    // a new vreg.
    unsigned VRBase = 0;

    if (!IsClone && !IsCloned)
      for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
           UI != E; ++UI) {
        SDNode *User = *UI;
        if (User->getOpcode() == ISD::CopyToReg && 
            User->getOperand(2).getNode() == Node &&
            User->getOperand(2).getResNo() == i) {
          unsigned Reg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
          if (TargetRegisterInfo::isVirtualRegister(Reg)) {
            VRBase = Reg;
            MI->addOperand(MachineOperand::CreateReg(Reg, true));
            break;
          }
        }
      }

    // Create the result registers for this node and add the result regs to
    // the machine instruction.
    if (VRBase == 0) {
      const TargetRegisterClass *RC = getInstrOperandRegClass(TRI, TII, II, i);
      assert(RC && "Isn't a register operand!");
      VRBase = MRI.createVirtualRegister(RC);
      MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    }

    SDValue Op(Node, i);
    if (IsClone)
      VRBaseMap.erase(Op);
    bool isNew = VRBaseMap.insert(std::make_pair(Op, VRBase)).second;
    isNew = isNew; // Silence compiler warning.
    assert(isNew && "Node emitted out of order - early");
  }
}

/// getVR - Return the virtual register corresponding to the specified result
/// of the specified node.
unsigned ScheduleDAGSDNodes::getVR(SDValue Op,
                                   DenseMap<SDValue, unsigned> &VRBaseMap) {
  if (Op.isMachineOpcode() &&
      Op.getMachineOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
    // Add an IMPLICIT_DEF instruction before every use.
    unsigned VReg = getDstOfOnlyCopyToRegUse(Op.getNode(), Op.getResNo());
    // IMPLICIT_DEF can produce any type of result so its TargetInstrDesc
    // does not include operand register class info.
    if (!VReg) {
      const TargetRegisterClass *RC = TLI->getRegClassFor(Op.getValueType());
      VReg = MRI.createVirtualRegister(RC);
    }
    BuildMI(BB, Op.getDebugLoc(), TII->get(TargetInstrInfo::IMPLICIT_DEF),VReg);
    return VReg;
  }

  DenseMap<SDValue, unsigned>::iterator I = VRBaseMap.find(Op);
  assert(I != VRBaseMap.end() && "Node emitted out of order - late");
  return I->second;
}


/// AddOperand - Add the specified operand to the specified machine instr.  II
/// specifies the instruction information for the node, and IIOpNum is the
/// operand number (in the II) that we are adding. IIOpNum and II are used for 
/// assertions only.
void ScheduleDAGSDNodes::AddOperand(MachineInstr *MI, SDValue Op,
                                    unsigned IIOpNum,
                                    const TargetInstrDesc *II,
                                    DenseMap<SDValue, unsigned> &VRBaseMap) {
  if (Op.isMachineOpcode()) {
    // Note that this case is redundant with the final else block, but we
    // include it because it is the most common and it makes the logic
    // simpler here.
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    // Get/emit the operand.
    unsigned VReg = getVR(Op, VRBaseMap);
    const TargetInstrDesc &TID = MI->getDesc();
    bool isOptDef = IIOpNum < TID.getNumOperands() &&
      TID.OpInfo[IIOpNum].isOptionalDef();
    MI->addOperand(MachineOperand::CreateReg(VReg, isOptDef));
    
    // Verify that it is right.
    assert(TargetRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
#ifndef NDEBUG
    if (II) {
      // There may be no register class for this operand if it is a variadic
      // argument (RC will be NULL in this case).  In this case, we just assume
      // the regclass is ok.
      const TargetRegisterClass *RC =
                          getInstrOperandRegClass(TRI, TII, *II, IIOpNum);
      assert((RC || II->isVariadic()) && "Expected reg class info!");
      const TargetRegisterClass *VRC = MRI.getRegClass(VReg);
      if (RC && VRC != RC) {
        cerr << "Register class of operand and regclass of use don't agree!\n";
        cerr << "Operand = " << IIOpNum << "\n";
        cerr << "Op->Val = "; Op.getNode()->dump(DAG); cerr << "\n";
        cerr << "MI = "; MI->print(cerr);
        cerr << "VReg = " << VReg << "\n";
        cerr << "VReg RegClass     size = " << VRC->getSize()
             << ", align = " << VRC->getAlignment() << "\n";
        cerr << "Expected RegClass size = " << RC->getSize()
             << ", align = " << RC->getAlignment() << "\n";
        cerr << "Fatal error, aborting.\n";
        abort();
      }
    }
#endif
  } else if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateImm(C->getZExtValue()));
  } else if (ConstantFPSDNode *F = dyn_cast<ConstantFPSDNode>(Op)) {
    const ConstantFP *CFP = F->getConstantFPValue();
    MI->addOperand(MachineOperand::CreateFPImm(CFP));
  } else if (RegisterSDNode *R = dyn_cast<RegisterSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateReg(R->getReg(), false));
  } else if (GlobalAddressSDNode *TGA = dyn_cast<GlobalAddressSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateGA(TGA->getGlobal(),TGA->getOffset()));
  } else if (BasicBlockSDNode *BB = dyn_cast<BasicBlockSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateMBB(BB->getBasicBlock()));
  } else if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateFI(FI->getIndex()));
  } else if (JumpTableSDNode *JT = dyn_cast<JumpTableSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateJTI(JT->getIndex()));
  } else if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Op)) {
    int Offset = CP->getOffset();
    unsigned Align = CP->getAlignment();
    const Type *Type = CP->getType();
    // MachineConstantPool wants an explicit alignment.
    if (Align == 0) {
      Align = TM.getTargetData()->getPreferredTypeAlignmentShift(Type);
      if (Align == 0) {
        // Alignment of vector types.  FIXME!
        Align = TM.getTargetData()->getTypePaddedSize(Type);
        Align = Log2_64(Align);
      }
    }
    
    unsigned Idx;
    if (CP->isMachineConstantPoolEntry())
      Idx = ConstPool->getConstantPoolIndex(CP->getMachineCPVal(), Align);
    else
      Idx = ConstPool->getConstantPoolIndex(CP->getConstVal(), Align);
    MI->addOperand(MachineOperand::CreateCPI(Idx, Offset));
  } else if (ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(Op)) {
    MI->addOperand(MachineOperand::CreateES(ES->getSymbol()));
  } else {
    assert(Op.getValueType() != MVT::Other &&
           Op.getValueType() != MVT::Flag &&
           "Chain and flag operands should occur at end of operand list!");
    unsigned VReg = getVR(Op, VRBaseMap);
    MI->addOperand(MachineOperand::CreateReg(VReg, false));
    
    // Verify that it is right.  Note that the reg class of the physreg and the
    // vreg don't necessarily need to match, but the target copy insertion has
    // to be able to handle it.  This handles things like copies from ST(0) to
    // an FP vreg on x86.
    assert(TargetRegisterInfo::isVirtualRegister(VReg) && "Not a vreg?");
    if (II && !II->isVariadic()) {
      assert(getInstrOperandRegClass(TRI, TII, *II, IIOpNum) &&
             "Don't have operand info for this instruction!");
    }
  }  
}

/// EmitSubregNode - Generate machine code for subreg nodes.
///
void ScheduleDAGSDNodes::EmitSubregNode(SDNode *Node, 
                                        DenseMap<SDValue, unsigned> &VRBaseMap) {
  unsigned VRBase = 0;
  unsigned Opc = Node->getMachineOpcode();
  
  // If the node is only used by a CopyToReg and the dest reg is a vreg, use
  // the CopyToReg'd destination register instead of creating a new vreg.
  for (SDNode::use_iterator UI = Node->use_begin(), E = Node->use_end();
       UI != E; ++UI) {
    SDNode *User = *UI;
    if (User->getOpcode() == ISD::CopyToReg && 
        User->getOperand(2).getNode() == Node) {
      unsigned DestReg = cast<RegisterSDNode>(User->getOperand(1))->getReg();
      if (TargetRegisterInfo::isVirtualRegister(DestReg)) {
        VRBase = DestReg;
        break;
      }
    }
  }
  
  if (Opc == TargetInstrInfo::EXTRACT_SUBREG) {
    unsigned SubIdx = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();

    // Create the extract_subreg machine instruction.
    MachineInstr *MI = BuildMI(MF, Node->getDebugLoc(),
                               TII->get(TargetInstrInfo::EXTRACT_SUBREG));

    // Figure out the register class to create for the destreg.
    const TargetRegisterClass *SRC = TLI->getRegClassFor(Node->getValueType(0));

    if (VRBase) {
      // Grab the destination register
#ifndef NDEBUG
      const TargetRegisterClass *DRC = MRI.getRegClass(VRBase);
      assert(SRC && DRC && SRC == DRC && 
             "Source subregister and destination must have the same class");
#endif
    } else {
      // Create the reg
      assert(SRC && "Couldn't find source register class");
      VRBase = MRI.createVirtualRegister(SRC);
    }
    
    // Add def, source, and subreg index
    MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    AddOperand(MI, Node->getOperand(0), 0, 0, VRBaseMap);
    MI->addOperand(MachineOperand::CreateImm(SubIdx));
    BB->insert(End, MI);
  } else if (Opc == TargetInstrInfo::INSERT_SUBREG ||
             Opc == TargetInstrInfo::SUBREG_TO_REG) {
    SDValue N0 = Node->getOperand(0);
    SDValue N1 = Node->getOperand(1);
    SDValue N2 = Node->getOperand(2);
    unsigned SubIdx = cast<ConstantSDNode>(N2)->getZExtValue();
    
      
    // Figure out the register class to create for the destreg.
    const TargetRegisterClass *TRC = 0;
    if (VRBase) {
      TRC = MRI.getRegClass(VRBase);
    } else {
      TRC = TLI->getRegClassFor(Node->getValueType(0));
      assert(TRC && "Couldn't determine register class for insert_subreg");
      VRBase = MRI.createVirtualRegister(TRC); // Create the reg
    }
    
    // Create the insert_subreg or subreg_to_reg machine instruction.
    MachineInstr *MI = BuildMI(MF, Node->getDebugLoc(), TII->get(Opc));
    MI->addOperand(MachineOperand::CreateReg(VRBase, true));
    
    // If creating a subreg_to_reg, then the first input operand
    // is an implicit value immediate, otherwise it's a register
    if (Opc == TargetInstrInfo::SUBREG_TO_REG) {
      const ConstantSDNode *SD = cast<ConstantSDNode>(N0);
      MI->addOperand(MachineOperand::CreateImm(SD->getZExtValue()));
    } else
      AddOperand(MI, N0, 0, 0, VRBaseMap);
    // Add the subregster being inserted
    AddOperand(MI, N1, 0, 0, VRBaseMap);
    MI->addOperand(MachineOperand::CreateImm(SubIdx));
    BB->insert(End, MI);
  } else
    assert(0 && "Node is not insert_subreg, extract_subreg, or subreg_to_reg");
     
  SDValue Op(Node, 0);
  bool isNew = VRBaseMap.insert(std::make_pair(Op, VRBase)).second;
  isNew = isNew; // Silence compiler warning.
  assert(isNew && "Node emitted out of order - early");
}

/// EmitNode - Generate machine code for an node and needed dependencies.
///
void ScheduleDAGSDNodes::EmitNode(SDNode *Node, bool IsClone, bool IsCloned,
                                  DenseMap<SDValue, unsigned> &VRBaseMap) {
  // If machine instruction
  if (Node->isMachineOpcode()) {
    unsigned Opc = Node->getMachineOpcode();
    
    // Handle subreg insert/extract specially
    if (Opc == TargetInstrInfo::EXTRACT_SUBREG || 
        Opc == TargetInstrInfo::INSERT_SUBREG ||
        Opc == TargetInstrInfo::SUBREG_TO_REG) {
      EmitSubregNode(Node, VRBaseMap);
      return;
    }

    if (Opc == TargetInstrInfo::IMPLICIT_DEF)
      // We want a unique VR for each IMPLICIT_DEF use.
      return;
    
    const TargetInstrDesc &II = TII->get(Opc);
    unsigned NumResults = CountResults(Node);
    unsigned NodeOperands = CountOperands(Node);
    unsigned MemOperandsEnd = ComputeMemOperandsEnd(Node);
    bool HasPhysRegOuts = (NumResults > II.getNumDefs()) &&
                          II.getImplicitDefs() != 0;
#ifndef NDEBUG
    unsigned NumMIOperands = NodeOperands + NumResults;
    assert((II.getNumOperands() == NumMIOperands ||
            HasPhysRegOuts || II.isVariadic()) &&
           "#operands for dag node doesn't match .td file!"); 
#endif

    // Create the new machine instruction.
    MachineInstr *MI = BuildMI(MF, Node->getDebugLoc(), II);
    
    // Add result register values for things that are defined by this
    // instruction.
    if (NumResults)
      CreateVirtualRegisters(Node, MI, II, IsClone, IsCloned, VRBaseMap);
    
    // Emit all of the actual operands of this instruction, adding them to the
    // instruction as appropriate.
    for (unsigned i = 0; i != NodeOperands; ++i)
      AddOperand(MI, Node->getOperand(i), i+II.getNumDefs(), &II, VRBaseMap);

    // Emit all of the memory operands of this instruction
    for (unsigned i = NodeOperands; i != MemOperandsEnd; ++i)
      AddMemOperand(MI, cast<MemOperandSDNode>(Node->getOperand(i))->MO);

    if (II.usesCustomDAGSchedInsertionHook()) {
      // Insert this instruction into the basic block using a target
      // specific inserter which may returns a new basic block.
      BB = TLI->EmitInstrWithCustomInserter(MI, BB);
      Begin = End = BB->end();
    } else {
      BB->insert(End, MI);
    }

    // Additional results must be an physical register def.
    if (HasPhysRegOuts) {
      for (unsigned i = II.getNumDefs(); i < NumResults; ++i) {
        unsigned Reg = II.getImplicitDefs()[i - II.getNumDefs()];
        if (Node->hasAnyUseOfValue(i))
          EmitCopyFromReg(Node, i, IsClone, IsCloned, Reg, VRBaseMap);
      }
    }
    return;
  }

  switch (Node->getOpcode()) {
  default:
#ifndef NDEBUG
    Node->dump(DAG);
#endif
    assert(0 && "This target-independent node should have been selected!");
    break;
  case ISD::EntryToken:
    assert(0 && "EntryToken should have been excluded from the schedule!");
    break;
  case ISD::TokenFactor: // fall thru
    break;
  case ISD::CopyToReg: {
    unsigned SrcReg;
    SDValue SrcVal = Node->getOperand(2);
    if (RegisterSDNode *R = dyn_cast<RegisterSDNode>(SrcVal))
      SrcReg = R->getReg();
    else
      SrcReg = getVR(SrcVal, VRBaseMap);
      
    unsigned DestReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    if (SrcReg == DestReg) // Coalesced away the copy? Ignore.
      break;
      
    const TargetRegisterClass *SrcTRC = 0, *DstTRC = 0;
    // Get the register classes of the src/dst.
    if (TargetRegisterInfo::isVirtualRegister(SrcReg))
      SrcTRC = MRI.getRegClass(SrcReg);
    else
      SrcTRC = TRI->getPhysicalRegisterRegClass(SrcReg,SrcVal.getValueType());

    if (TargetRegisterInfo::isVirtualRegister(DestReg))
      DstTRC = MRI.getRegClass(DestReg);
    else
      DstTRC = TRI->getPhysicalRegisterRegClass(DestReg,
                                            Node->getOperand(1).getValueType());
    TII->copyRegToReg(*BB, End, DestReg, SrcReg, DstTRC, SrcTRC);
    break;
  }
  case ISD::CopyFromReg: {
    unsigned SrcReg = cast<RegisterSDNode>(Node->getOperand(1))->getReg();
    EmitCopyFromReg(Node, 0, IsClone, IsCloned, SrcReg, VRBaseMap);
    break;
  }
  case ISD::INLINEASM: {
    unsigned NumOps = Node->getNumOperands();
    if (Node->getOperand(NumOps-1).getValueType() == MVT::Flag)
      --NumOps;  // Ignore the flag operand.
      
    // Create the inline asm machine instruction.
    MachineInstr *MI = BuildMI(MF, Node->getDebugLoc(),
                               TII->get(TargetInstrInfo::INLINEASM));

    // Add the asm string as an external symbol operand.
    const char *AsmStr =
      cast<ExternalSymbolSDNode>(Node->getOperand(1))->getSymbol();
    MI->addOperand(MachineOperand::CreateES(AsmStr));
      
    // Add all of the operand registers to the instruction.
    for (unsigned i = 2; i != NumOps;) {
      unsigned Flags =
        cast<ConstantSDNode>(Node->getOperand(i))->getZExtValue();
      unsigned NumVals = Flags >> 3;
        
      MI->addOperand(MachineOperand::CreateImm(Flags));
      ++i;  // Skip the ID value.
        
      switch (Flags & 7) {
      default: assert(0 && "Bad flags!");
      case 2:   // Def of register.
        for (; NumVals; --NumVals, ++i) {
          unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
          MI->addOperand(MachineOperand::CreateReg(Reg, true));
        }
        break;
      case 6:   // Def of earlyclobber register.
        for (; NumVals; --NumVals, ++i) {
          unsigned Reg = cast<RegisterSDNode>(Node->getOperand(i))->getReg();
          MI->addOperand(MachineOperand::CreateReg(Reg, true, false, false, 
                                                   false, 0, true));
        }
        break;
      case 1:  // Use of register.
      case 3:  // Immediate.
      case 4:  // Addressing mode.
        // The addressing mode has been selected, just add all of the
        // operands to the machine instruction.
        for (; NumVals; --NumVals, ++i)
          AddOperand(MI, Node->getOperand(i), 0, 0, VRBaseMap);
        break;
      }
    }
    BB->insert(End, MI);
    break;
  }
  }
}

/// EmitSchedule - Emit the machine code in scheduled order.
MachineBasicBlock *ScheduleDAGSDNodes::EmitSchedule() {
  DenseMap<SDValue, unsigned> VRBaseMap;
  DenseMap<SUnit*, unsigned> CopyVRBaseMap;
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    SUnit *SU = Sequence[i];
    if (!SU) {
      // Null SUnit* is a noop.
      EmitNoop();
      continue;
    }

    // For pre-regalloc scheduling, create instructions corresponding to the
    // SDNode and any flagged SDNodes and append them to the block.
    if (!SU->getNode()) {
      // Emit a copy.
      EmitPhysRegCopy(SU, CopyVRBaseMap);
      continue;
    }

    SmallVector<SDNode *, 4> FlaggedNodes;
    for (SDNode *N = SU->getNode()->getFlaggedNode(); N;
         N = N->getFlaggedNode())
      FlaggedNodes.push_back(N);
    while (!FlaggedNodes.empty()) {
      EmitNode(FlaggedNodes.back(), SU->OrigNode != SU, SU->isCloned,VRBaseMap);
      FlaggedNodes.pop_back();
    }
    EmitNode(SU->getNode(), SU->OrigNode != SU, SU->isCloned, VRBaseMap);
  }

  return BB;
}
