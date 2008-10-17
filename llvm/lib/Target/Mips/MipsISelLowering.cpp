//===-- MipsISelLowering.cpp - Mips DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Mips uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "mips-lower"

#include "MipsISelLowering.h"
#include "MipsMachineFunction.h"
#include "MipsTargetMachine.h"
#include "MipsSubtarget.h"
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
#include <queue>
#include <set>

using namespace llvm;

const char *MipsTargetLowering::
getTargetNodeName(unsigned Opcode) const 
{
  switch (Opcode) 
  {
    case MipsISD::JmpLink    : return "MipsISD::JmpLink";
    case MipsISD::Hi         : return "MipsISD::Hi";
    case MipsISD::Lo         : return "MipsISD::Lo";
    case MipsISD::GPRel      : return "MipsISD::GPRel";
    case MipsISD::Ret        : return "MipsISD::Ret";
    case MipsISD::CMov       : return "MipsISD::CMov";
    case MipsISD::SelectCC   : return "MipsISD::SelectCC";
    case MipsISD::FPSelectCC : return "MipsISD::FPSelectCC";
    case MipsISD::FPBrcond   : return "MipsISD::FPBrcond";
    case MipsISD::FPCmp      : return "MipsISD::FPCmp";
    default                  : return NULL;
  }
}

MipsTargetLowering::
MipsTargetLowering(MipsTargetMachine &TM): TargetLowering(TM) 
{
  Subtarget = &TM.getSubtarget<MipsSubtarget>();

  // Mips does not have i1 type, so use i32 for
  // setcc operations results (slt, sgt, ...). 
  setSetCCResultContents(ZeroOrOneSetCCResult);

  // JumpTable targets must use GOT when using PIC_
  setUsesGlobalOffsetTable(true);

  // Set up the register classes
  addRegisterClass(MVT::i32, Mips::CPURegsRegisterClass);

  // When dealing with single precision only, use libcalls
  if (!Subtarget->isSingleFloat()) {
    addRegisterClass(MVT::f32, Mips::AFGR32RegisterClass);
    if (!Subtarget->isFP64bit())
      addRegisterClass(MVT::f64, Mips::AFGR64RegisterClass);
  } else 
    addRegisterClass(MVT::f32, Mips::FGR32RegisterClass);

  // Legal fp constants
  addLegalFPImmediate(APFloat(+0.0f));

  // Load extented operations for i1 types must be promoted 
  setLoadXAction(ISD::EXTLOAD,  MVT::i1,  Promote);
  setLoadXAction(ISD::ZEXTLOAD, MVT::i1,  Promote);
  setLoadXAction(ISD::SEXTLOAD, MVT::i1,  Promote);

  // Used by legalize types to correctly generate the setcc result. 
  // Without this, every float setcc comes with a AND/OR with the result, 
  // we don't want this, since the fpcmp result goes to a flag register, 
  // which is used implicitly by brcond and select operations.
  AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);

  // Mips Custom Operations
  setOperationAction(ISD::GlobalAddress,      MVT::i32,   Custom);
  setOperationAction(ISD::GlobalTLSAddress,   MVT::i32,   Custom);
  setOperationAction(ISD::RET,                MVT::Other, Custom);
  setOperationAction(ISD::JumpTable,          MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,       MVT::i32,   Custom);
  setOperationAction(ISD::SELECT,             MVT::f32,   Custom);
  setOperationAction(ISD::SELECT,             MVT::i32,   Custom);
  setOperationAction(ISD::SETCC,              MVT::f32,   Custom);
  setOperationAction(ISD::BRCOND,             MVT::Other, Custom);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32,   Custom);

  // We custom lower AND/OR to handle the case where the DAG contain 'ands/ors' 
  // with operands comming from setcc fp comparions. This is necessary since 
  // the result from these setcc are in a flag registers (FCR31).
  setOperationAction(ISD::AND,              MVT::i32,   Custom);
  setOperationAction(ISD::OR,               MVT::i32,   Custom);

  // Operations not directly supported by Mips.
  setOperationAction(ISD::BR_JT,             MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,             MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC,         MVT::Other, Expand);
  setOperationAction(ISD::UINT_TO_FP,        MVT::i32,   Expand);
  setOperationAction(ISD::FP_TO_UINT,        MVT::i32,   Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1,    Expand);
  setOperationAction(ISD::CTPOP,             MVT::i32,   Expand);
  setOperationAction(ISD::CTTZ,              MVT::i32,   Expand);
  setOperationAction(ISD::ROTL,              MVT::i32,   Expand);
  setOperationAction(ISD::SHL_PARTS,         MVT::i32,   Expand);
  setOperationAction(ISD::SRA_PARTS,         MVT::i32,   Expand);
  setOperationAction(ISD::SRL_PARTS,         MVT::i32,   Expand);
  setOperationAction(ISD::FCOPYSIGN,         MVT::f32,   Expand);

  // We don't have line number support yet.
  setOperationAction(ISD::DBG_STOPPOINT,     MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC,         MVT::Other, Expand);
  setOperationAction(ISD::DBG_LABEL,         MVT::Other, Expand);
  setOperationAction(ISD::EH_LABEL,          MVT::Other, Expand);

  // Use the default for now
  setOperationAction(ISD::STACKSAVE,         MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE,      MVT::Other, Expand);
  setOperationAction(ISD::MEMBARRIER,        MVT::Other, Expand);

  if (Subtarget->isSingleFloat())
    setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);

  if (!Subtarget->hasSEInReg()) {
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8,  Expand);
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  }

  if (!Subtarget->hasBitCount())
    setOperationAction(ISD::CTLZ, MVT::i32, Expand);

  if (!Subtarget->hasSwap())
    setOperationAction(ISD::BSWAP, MVT::i32, Expand);

  setStackPointerRegisterToSaveRestore(Mips::SP);
  computeRegisterProperties();
}


MVT MipsTargetLowering::getSetCCResultType(const SDValue &) const {
  return MVT::i32;
}


SDValue MipsTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) 
{
  switch (Op.getOpcode()) 
  {
    case ISD::AND:                return LowerANDOR(Op, DAG);
    case ISD::BRCOND:             return LowerBRCOND(Op, DAG);
    case ISD::CALL:               return LowerCALL(Op, DAG);
    case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
    case ISD::DYNAMIC_STACKALLOC: return LowerDYNAMIC_STACKALLOC(Op, DAG);
    case ISD::FORMAL_ARGUMENTS:   return LowerFORMAL_ARGUMENTS(Op, DAG);
    case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
    case ISD::GlobalTLSAddress:   return LowerGlobalTLSAddress(Op, DAG);
    case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
    case ISD::OR:                 return LowerANDOR(Op, DAG);
    case ISD::RET:                return LowerRET(Op, DAG);
    case ISD::SELECT:             return LowerSELECT(Op, DAG);
    case ISD::SETCC:              return LowerSETCC(Op, DAG);
  }
  return SDValue();
}

//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//

// AddLiveIn - This helper function adds the specified physical register to the
// MachineFunction as a live in value.  It also creates a corresponding
// virtual register for it.
static unsigned
AddLiveIn(MachineFunction &MF, unsigned PReg, TargetRegisterClass *RC) 
{
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getRegInfo().createVirtualRegister(RC);
  MF.getRegInfo().addLiveIn(PReg, VReg);
  return VReg;
}

// A address must be loaded from a small section if its size is less than the 
// small section size threshold. Data in this section must be addressed using 
// gp_rel operator.
bool MipsTargetLowering::IsInSmallSection(unsigned Size) {
  return (Size > 0 && (Size <= Subtarget->getSSectionThreshold()));
}

// Discover if this global address can be placed into small data/bss section. 
bool MipsTargetLowering::IsGlobalInSmallSection(GlobalValue *GV)
{
  const TargetData *TD = getTargetData();
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GV);

  if (!GVA)
    return false;
  
  const Type *Ty = GV->getType()->getElementType();
  unsigned Size = TD->getABITypeSize(Ty);

  // if this is a internal constant string, there is a special
  // section for it, but not in small data/bss.
  if (GVA->hasInitializer() && GV->hasInternalLinkage()) {
    Constant *C = GVA->getInitializer();
    const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
    if (CVA && CVA->isCString()) 
      return false;
  }

  return IsInSmallSection(Size);
}

// Get fp branch code (not opcode) from condition code.
static Mips::FPBranchCode GetFPBranchCodeFromCond(Mips::CondCode CC) {
  if (CC >= Mips::FCOND_F && CC <= Mips::FCOND_NGT)
    return Mips::BRANCH_T;

  if (CC >= Mips::FCOND_T && CC <= Mips::FCOND_GT)
    return Mips::BRANCH_F;

  return Mips::BRANCH_INVALID;
}
  
static unsigned FPBranchCodeToOpc(Mips::FPBranchCode BC) {
  switch(BC) {
    default:
      assert(0 && "Unknown branch code");
    case Mips::BRANCH_T  : return Mips::BC1T;
    case Mips::BRANCH_F  : return Mips::BC1F;
    case Mips::BRANCH_TL : return Mips::BC1TL;
    case Mips::BRANCH_FL : return Mips::BC1FL;
  }
}

static Mips::CondCode FPCondCCodeToFCC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown fp condition code!");
  case ISD::SETEQ:  
  case ISD::SETOEQ: return Mips::FCOND_EQ;
  case ISD::SETUNE: return Mips::FCOND_OGL;
  case ISD::SETLT:  
  case ISD::SETOLT: return Mips::FCOND_OLT;
  case ISD::SETGT:  
  case ISD::SETOGT: return Mips::FCOND_OGT;
  case ISD::SETLE:  
  case ISD::SETOLE: return Mips::FCOND_OLE; 
  case ISD::SETGE:
  case ISD::SETOGE: return Mips::FCOND_OGE;
  case ISD::SETULT: return Mips::FCOND_ULT;
  case ISD::SETULE: return Mips::FCOND_ULE; 
  case ISD::SETUGT: return Mips::FCOND_UGT;
  case ISD::SETUGE: return Mips::FCOND_UGE;
  case ISD::SETUO:  return Mips::FCOND_UN; 
  case ISD::SETO:   return Mips::FCOND_OR;
  case ISD::SETNE:  
  case ISD::SETONE: return Mips::FCOND_NEQ;
  case ISD::SETUEQ: return Mips::FCOND_UEQ;
  }
}

MachineBasicBlock *
MipsTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                MachineBasicBlock *BB) 
{
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  bool isFPCmp = false;

  switch (MI->getOpcode()) {
  default: assert(false && "Unexpected instr type to insert");
  case Mips::Select_FCC:
  case Mips::Select_FCC_SO32:
  case Mips::Select_FCC_AS32:
  case Mips::Select_FCC_D32:
    isFPCmp = true; // FALL THROUGH
  case Mips::Select_CC:
  case Mips::Select_CC_SO32:
  case Mips::Select_CC_AS32:
  case Mips::Select_CC_D32: {
    // To "insert" a SELECT_CC instruction, we actually have to insert the
    // diamond control-flow pattern.  The incoming instruction knows the
    // destination vreg to set, the condition code register to branch on, the
    // true/false values to select between, and a branch opcode to use.
    const BasicBlock *LLVM_BB = BB->getBasicBlock();
    MachineFunction::iterator It = BB;
    ++It;

    //  thisMBB:
    //  ...
    //   TrueVal = ...
    //   setcc r1, r2, r3
    //   bNE   r1, r0, copy1MBB
    //   fallthrough --> copy0MBB
    MachineBasicBlock *thisMBB  = BB;
    MachineFunction *F = BB->getParent();
    MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
    MachineBasicBlock *sinkMBB  = F->CreateMachineBasicBlock(LLVM_BB);

    // Emit the right instruction according to the type of the operands compared
    if (isFPCmp) {
      // Find the condiction code present in the setcc operation.
      Mips::CondCode CC = (Mips::CondCode)MI->getOperand(4).getImm();
      // Get the branch opcode from the branch code.
      unsigned Opc = FPBranchCodeToOpc(GetFPBranchCodeFromCond(CC));
      BuildMI(BB, TII->get(Opc)).addMBB(sinkMBB);
    } else
      BuildMI(BB, TII->get(Mips::BNE)).addReg(MI->getOperand(1).getReg())
        .addReg(Mips::ZERO).addMBB(sinkMBB);

    F->insert(It, copy0MBB);
    F->insert(It, sinkMBB);
    // Update machine-CFG edges by first adding all successors of the current
    // block to the new block which will contain the Phi node for the select.
    for(MachineBasicBlock::succ_iterator i = BB->succ_begin(),
        e = BB->succ_end(); i != e; ++i)
      sinkMBB->addSuccessor(*i);
    // Next, remove all successors of the current block, and add the true
    // and fallthrough blocks as its successors.
    while(!BB->succ_empty())
      BB->removeSuccessor(BB->succ_begin());
    BB->addSuccessor(copy0MBB);
    BB->addSuccessor(sinkMBB);

    //  copy0MBB:
    //   %FalseValue = ...
    //   # fallthrough to sinkMBB
    BB = copy0MBB;

    // Update machine-CFG edges
    BB->addSuccessor(sinkMBB);

    //  sinkMBB:
    //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
    //  ...
    BB = sinkMBB;
    BuildMI(BB, TII->get(Mips::PHI), MI->getOperand(0).getReg())
      .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
      .addReg(MI->getOperand(3).getReg()).addMBB(thisMBB);

    F->DeleteMachineInstr(MI);   // The pseudo instruction is gone now.
    return BB;
  }
  }
}

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//

SDValue MipsTargetLowering::
LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG)
{
  SDValue Chain = Op.getOperand(0);
  SDValue Size = Op.getOperand(1);

  // Get a reference from Mips stack pointer
  SDValue StackPointer = DAG.getCopyFromReg(Chain, Mips::SP, MVT::i32);

  // Subtract the dynamic size from the actual stack size to
  // obtain the new stack size.
  SDValue Sub = DAG.getNode(ISD::SUB, MVT::i32, StackPointer, Size);

  // The Sub result contains the new stack start address, so it 
  // must be placed in the stack pointer register.
  Chain = DAG.getCopyToReg(StackPointer.getValue(1), Mips::SP, Sub);
  
  // This node always has two return values: a new stack pointer 
  // value and a chain
  SDValue Ops[2] = { Sub, Chain };
  return DAG.getMergeValues(Ops, 2);
}

SDValue MipsTargetLowering::
LowerANDOR(SDValue Op, SelectionDAG &DAG)
{
  SDValue LHS   = Op.getOperand(0);
  SDValue RHS   = Op.getOperand(1);
  
  if (LHS.getOpcode() != MipsISD::FPCmp || RHS.getOpcode() != MipsISD::FPCmp)
    return Op;

  SDValue True  = DAG.getConstant(1, MVT::i32);
  SDValue False = DAG.getConstant(0, MVT::i32);

  SDValue LSEL = DAG.getNode(MipsISD::FPSelectCC, True.getValueType(), 
                             LHS, True, False, LHS.getOperand(2));
  SDValue RSEL = DAG.getNode(MipsISD::FPSelectCC, True.getValueType(), 
                             RHS, True, False, RHS.getOperand(2));

  return DAG.getNode(Op.getOpcode(), MVT::i32, LSEL, RSEL);
}

SDValue MipsTargetLowering::
LowerBRCOND(SDValue Op, SelectionDAG &DAG)
{
  // The first operand is the chain, the second is the condition, the third is 
  // the block to branch to if the condition is true.
  SDValue Chain = Op.getOperand(0);
  SDValue Dest = Op.getOperand(2);

  if (Op.getOperand(1).getOpcode() != MipsISD::FPCmp)
    return Op;
  
  SDValue CondRes = Op.getOperand(1);
  SDValue CCNode  = CondRes.getOperand(2);
  Mips::CondCode CC =
    (Mips::CondCode)cast<ConstantSDNode>(CCNode)->getZExtValue();
  SDValue BrCode = DAG.getConstant(GetFPBranchCodeFromCond(CC), MVT::i32); 

  return DAG.getNode(MipsISD::FPBrcond, Op.getValueType(), Chain, BrCode, 
             Dest, CondRes);
}

SDValue MipsTargetLowering::
LowerSETCC(SDValue Op, SelectionDAG &DAG)
{
  // The operands to this are the left and right operands to compare (ops #0, 
  // and #1) and the condition code to compare them with (op #2) as a 
  // CondCodeSDNode.
  SDValue LHS = Op.getOperand(0); 
  SDValue RHS = Op.getOperand(1); 

  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  
  return DAG.getNode(MipsISD::FPCmp, Op.getValueType(), LHS, RHS, 
                 DAG.getConstant(FPCondCCodeToFCC(CC), MVT::i32));
}

SDValue MipsTargetLowering::
LowerSELECT(SDValue Op, SelectionDAG &DAG) 
{
  SDValue Cond  = Op.getOperand(0); 
  SDValue True  = Op.getOperand(1);
  SDValue False = Op.getOperand(2);

  // if the incomming condition comes from a integer compare, the select 
  // operation must be SelectCC or a conditional move if the subtarget 
  // supports it.
  if (Cond.getOpcode() != MipsISD::FPCmp) {
    if (Subtarget->hasCondMov() && !True.getValueType().isFloatingPoint())
      return Op;
    return DAG.getNode(MipsISD::SelectCC, True.getValueType(), 
                       Cond, True, False);
  }

  // if the incomming condition comes from fpcmp, the select
  // operation must use FPSelectCC.
  SDValue CCNode = Cond.getOperand(2);
  return DAG.getNode(MipsISD::FPSelectCC, True.getValueType(), 
                     Cond, True, False, CCNode);
}

SDValue MipsTargetLowering::
LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) 
{
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDValue GA = DAG.getTargetGlobalAddress(GV, MVT::i32);

  if (!Subtarget->hasABICall()) {
    const MVT *VTs = DAG.getNodeValueTypes(MVT::i32);
    SDValue Ops[] = { GA };
    // %gp_rel relocation
    if (!isa<Function>(GV) && IsGlobalInSmallSection(GV)) { 
      SDValue GPRelNode = DAG.getNode(MipsISD::GPRel, VTs, 1, Ops, 1);
      SDValue GOT = DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, MVT::i32);
      return DAG.getNode(ISD::ADD, MVT::i32, GOT, GPRelNode); 
    }
    // %hi/%lo relocation
    SDValue HiPart = DAG.getNode(MipsISD::Hi, VTs, 1, Ops, 1);
    SDValue Lo = DAG.getNode(MipsISD::Lo, MVT::i32, GA);
    return DAG.getNode(ISD::ADD, MVT::i32, HiPart, Lo);

  } else { // Abicall relocations, TODO: make this cleaner.
    SDValue ResNode = DAG.getLoad(MVT::i32, DAG.getEntryNode(), GA, NULL, 0);
    // On functions and global targets not internal linked only
    // a load from got/GP is necessary for PIC to work.
    if (!GV->hasInternalLinkage() || isa<Function>(GV))
      return ResNode;
    SDValue Lo = DAG.getNode(MipsISD::Lo, MVT::i32, GA);
    return DAG.getNode(ISD::ADD, MVT::i32, ResNode, Lo);
  }

  assert(0 && "Dont know how to handle GlobalAddress");
  return SDValue(0,0);
}

SDValue MipsTargetLowering::
LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG)
{
  assert(0 && "TLS not implemented for MIPS.");
  return SDValue(); // Not reached
}

SDValue MipsTargetLowering::
LowerJumpTable(SDValue Op, SelectionDAG &DAG) 
{
  SDValue ResNode;
  SDValue HiPart; 

  MVT PtrVT = Op.getValueType();
  JumpTableSDNode *JT  = cast<JumpTableSDNode>(Op);
  SDValue JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT);

  if (getTargetMachine().getRelocationModel() != Reloc::PIC_) {
    const MVT *VTs = DAG.getNodeValueTypes(MVT::i32);
    SDValue Ops[] = { JTI };
    HiPart = DAG.getNode(MipsISD::Hi, VTs, 1, Ops, 1);
  } else // Emit Load from Global Pointer
    HiPart = DAG.getLoad(MVT::i32, DAG.getEntryNode(), JTI, NULL, 0);

  SDValue Lo = DAG.getNode(MipsISD::Lo, MVT::i32, JTI);
  ResNode = DAG.getNode(ISD::ADD, MVT::i32, HiPart, Lo);

  return ResNode;
}

SDValue MipsTargetLowering::
LowerConstantPool(SDValue Op, SelectionDAG &DAG) 
{
  SDValue ResNode;
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);
  Constant *C = N->getConstVal();
  SDValue CP = DAG.getTargetConstantPool(C, MVT::i32, N->getAlignment());

  // gp_rel relocation
  // FIXME: we should reference the constant pool using small data sections, 
  // but the asm printer currently doens't support this feature without
  // hacking it. This feature should come soon so we can uncomment the 
  // stuff below.
  //if (!Subtarget->hasABICall() &&  
  //    IsInSmallSection(getTargetData()->getABITypeSize(C->getType()))) {
  //  SDValue GPRelNode = DAG.getNode(MipsISD::GPRel, MVT::i32, CP);
  //  SDValue GOT = DAG.getNode(ISD::GLOBAL_OFFSET_TABLE, MVT::i32);
  //  ResNode = DAG.getNode(ISD::ADD, MVT::i32, GOT, GPRelNode); 
  //} else { // %hi/%lo relocation
    SDValue HiPart = DAG.getNode(MipsISD::Hi, MVT::i32, CP);
    SDValue Lo = DAG.getNode(MipsISD::Lo, MVT::i32, CP);
    ResNode = DAG.getNode(ISD::ADD, MVT::i32, HiPart, Lo);
  //}

  return ResNode;
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//
//  The lower operations present on calling convention works on this order:
//      LowerCALL (virt regs --> phys regs, virt regs --> stack) 
//      LowerFORMAL_ARGUMENTS (phys --> virt regs, stack --> virt regs)
//      LowerRET (virt regs --> phys regs)
//      LowerCALL (phys regs --> virt regs)
//
//===----------------------------------------------------------------------===//

#include "MipsGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//                  CALL Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// LowerCCCCallTo - functions arguments are copied from virtual
/// regs to (physical regs)/(stack frame), CALLSEQ_START and
/// CALLSEQ_END are emitted.
/// TODO: isVarArg, isTailCall.
SDValue MipsTargetLowering::
LowerCALL(SDValue Op, SelectionDAG &DAG)
{
  MachineFunction &MF = DAG.getMachineFunction();

  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  SDValue Chain = TheCall->getChain();
  SDValue Callee = TheCall->getCallee();
  bool isVarArg = TheCall->isVarArg();
  unsigned CC = TheCall->getCallingConv();

  MachineFrameInfo *MFI = MF.getFrameInfo();

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);

  // To meet O32 ABI, Mips must always allocate 16 bytes on
  // the stack (even if less than 4 are used as arguments)
  if (Subtarget->isABI_O32()) {
    int VTsize = MVT(MVT::i32).getSizeInBits()/8;
    MFI->CreateFixedObject(VTsize, (VTsize*3));
  }

  CCInfo.AnalyzeCallOperands(TheCall, CC_Mips);
  
  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));

  // With EABI is it possible to have 16 args on registers.
  SmallVector<std::pair<unsigned, SDValue>, 16> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // First/LastArgStackLoc contains the first/last 
  // "at stack" argument location.
  int LastArgStackLoc = 0;
  unsigned FirstStackArgLoc = (Subtarget->isABI_EABI() ? 0 : 16);

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    // Arguments start after the 5 first operands of ISD::CALL
    SDValue Arg = TheCall->getArg(i);
    
    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default: assert(0 && "Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, VA.getLocVT(), Arg);
      break;
    }
    
    // Arguments that can be passed on register must be kept at 
    // RegsToPass vector
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      continue;
    }
    
    // Register cant get to this point...
    assert(VA.isMemLoc());
    
    // Create the frame index object for this incoming parameter
    // This guarantees that when allocating Local Area the firsts
    // 16 bytes which are alwayes reserved won't be overwritten
    // if O32 ABI is used. For EABI the first address is zero.
    LastArgStackLoc = (FirstStackArgLoc + VA.getLocMemOffset());
    int FI = MFI->CreateFixedObject(VA.getValVT().getSizeInBits()/8,
                                    LastArgStackLoc);

    SDValue PtrOff = DAG.getFrameIndex(FI,getPointerTy());

    // emit ISD::STORE whichs stores the 
    // parameter value to a stack Location
    MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
  }

  // Transform all store nodes into one single node because all store
  // nodes are independent of each other.
  if (!MemOpChains.empty())     
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, 
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token 
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emited instructions must be
  // stuck together.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, 
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol 
  // node so that legalize doesn't hack it. 
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) 
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());


  // MipsJmpLink = #chain, #target_address, #opt_in_flags...
  //             = Chain, Callee, Reg#1, Reg#2, ...  
  //
  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are 
  // known live into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  Chain  = DAG.getNode(MipsISD::JmpLink, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  InFlag = Chain.getValue(1);

  // Create a stack location to hold GP when PIC is used. This stack 
  // location is used on function prologue to save GP and also after all 
  // emited CALL's to restore GP. 
  if (getTargetMachine().getRelocationModel() == Reloc::PIC_) {
      // Function can have an arbitrary number of calls, so 
      // hold the LastArgStackLoc with the biggest offset.
      int FI;
      MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
      if (LastArgStackLoc >= MipsFI->getGPStackOffset()) {
        LastArgStackLoc = (!LastArgStackLoc) ? (16) : (LastArgStackLoc+4);
        // Create the frame index only once. SPOffset here can be anything 
        // (this will be fixed on processFunctionBeforeFrameFinalized)
        if (MipsFI->getGPStackOffset() == -1) {
          FI = MFI->CreateFixedObject(4, 0);
          MipsFI->setGPFI(FI);
        }
        MipsFI->setGPStackOffset(LastArgStackLoc);
      }

      // Reload GP value.
      FI = MipsFI->getGPFI();
      SDValue FIN = DAG.getFrameIndex(FI,getPointerTy());
      SDValue GPLoad = DAG.getLoad(MVT::i32, Chain, FIN, NULL, 0);
      Chain = GPLoad.getValue(1);
      Chain = DAG.getCopyToReg(Chain, DAG.getRegister(Mips::GP, MVT::i32), 
                               GPLoad, SDValue(0,0));
      InFlag = Chain.getValue(1);
  }      

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return SDValue(LowerCallResult(Chain, InFlag, TheCall, CC, DAG), Op.getResNo());
}

/// LowerCallResult - Lower the result values of an ISD::CALL into the
/// appropriate copies out of appropriate physical registers.  This assumes that
/// Chain/InFlag are the input chain/flag to use, and that TheCall is the call
/// being lowered. Returns a SDNode with the same number of values as the 
/// ISD::CALL.
SDNode *MipsTargetLowering::
LowerCallResult(SDValue Chain, SDValue InFlag, CallSDNode *TheCall, 
        unsigned CallingConv, SelectionDAG &DAG) {
  
  bool isVarArg = TheCall->isVarArg();

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallingConv, isVarArg, getTargetMachine(), RVLocs);

  CCInfo.AnalyzeCallResult(TheCall, RetCC_Mips);
  SmallVector<SDValue, 8> ResultVals;

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    Chain = DAG.getCopyFromReg(Chain, RVLocs[i].getLocReg(),
                                 RVLocs[i].getValVT(), InFlag).getValue(1);
    InFlag = Chain.getValue(2);
    ResultVals.push_back(Chain.getValue(0));
  }
  
  ResultVals.push_back(Chain);

  // Merge everything together with a MERGE_VALUES node.
  return DAG.getMergeValues(TheCall->getVTList(), &ResultVals[0],
                            ResultVals.size()).getNode();
}

//===----------------------------------------------------------------------===//
//             FORMAL_ARGUMENTS Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// LowerFORMAL_ARGUMENTS - transform physical registers into
/// virtual registers and generate load operations for
/// arguments places on the stack.
/// TODO: isVarArg
SDValue MipsTargetLowering::
LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG) 
{
  SDValue Root = Op.getOperand(0);
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue() != 0;
  unsigned CC = DAG.getMachineFunction().getFunction()->getCallingConv();

  unsigned StackReg = MF.getTarget().getRegisterInfo()->getFrameRegister(MF);

  // GP must be live into PIC and non-PIC call target.
  AddLiveIn(MF, Mips::GP, Mips::CPURegsRegisterClass);

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);

  CCInfo.AnalyzeFormalArguments(Op.getNode(), CC_Mips);
  SmallVector<SDValue, 16> ArgValues;
  SDValue StackPtr;

  unsigned FirstStackArgLoc = (Subtarget->isABI_EABI() ? 0 : 16);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {

    CCValAssign &VA = ArgLocs[i];

    // Arguments stored on registers
    if (VA.isRegLoc()) {
      MVT RegVT = VA.getLocVT();
      TargetRegisterClass *RC = 0;
            
      if (RegVT == MVT::i32)
        RC = Mips::CPURegsRegisterClass; 
      else if (RegVT == MVT::f32) {
        if (Subtarget->isSingleFloat())
          RC = Mips::FGR32RegisterClass;
        else
          RC = Mips::AFGR32RegisterClass;
      } else if (RegVT == MVT::f64) {
        if (!Subtarget->isSingleFloat()) 
          RC = Mips::AFGR64RegisterClass;
      } else  
        assert(0 && "RegVT not supported by FORMAL_ARGUMENTS Lowering");

      // Transform the arguments stored on 
      // physical registers into virtual ones
      unsigned Reg = AddLiveIn(DAG.getMachineFunction(), VA.getLocReg(), RC);
      SDValue ArgValue = DAG.getCopyFromReg(Root, Reg, RegVT);
      
      // If this is an 8 or 16-bit value, it is really passed promoted 
      // to 32 bits.  Insert an assert[sz]ext to capture this, then 
      // truncate to the right size.
      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      
      if (VA.getLocInfo() != CCValAssign::Full)
        ArgValue = DAG.getNode(ISD::TRUNCATE, VA.getValVT(), ArgValue);

      ArgValues.push_back(ArgValue);

      // To meet ABI, when VARARGS are passed on registers, the registers
      // must have their values written to the caller stack frame. 
      if ((isVarArg) && (Subtarget->isABI_O32())) {
        if (StackPtr.getNode() == 0)
          StackPtr = DAG.getRegister(StackReg, getPointerTy());
     
        // The stack pointer offset is relative to the caller stack frame. 
        // Since the real stack size is unknown here, a negative SPOffset 
        // is used so there's a way to adjust these offsets when the stack
        // size get known (on EliminateFrameIndex). A dummy SPOffset is 
        // used instead of a direct negative address (which is recorded to
        // be used on emitPrologue) to avoid mis-calc of the first stack 
        // offset on PEI::calculateFrameObjectOffsets.
        // Arguments are always 32-bit.
        int FI = MFI->CreateFixedObject(4, 0);
        MipsFI->recordStoreVarArgsFI(FI, -(4+(i*4)));
        SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy());
      
        // emit ISD::STORE whichs stores the 
        // parameter value to a stack Location
        ArgValues.push_back(DAG.getStore(Root, ArgValue, PtrOff, NULL, 0));
      }

    } else { // VA.isRegLoc()

      // sanity check
      assert(VA.isMemLoc());
      
      // The stack pointer offset is relative to the caller stack frame. 
      // Since the real stack size is unknown here, a negative SPOffset 
      // is used so there's a way to adjust these offsets when the stack
      // size get known (on EliminateFrameIndex). A dummy SPOffset is 
      // used instead of a direct negative address (which is recorded to
      // be used on emitPrologue) to avoid mis-calc of the first stack 
      // offset on PEI::calculateFrameObjectOffsets.
      // Arguments are always 32-bit.
      unsigned ArgSize = VA.getLocVT().getSizeInBits()/8;
      int FI = MFI->CreateFixedObject(ArgSize, 0);
      MipsFI->recordLoadArgsFI(FI, -(ArgSize+
        (FirstStackArgLoc + VA.getLocMemOffset())));

      // Create load nodes to retrieve arguments from the stack
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
      ArgValues.push_back(DAG.getLoad(VA.getValVT(), Root, FIN, NULL, 0));
    }
  }

  // The mips ABIs for returning structs by value requires that we copy
  // the sret argument into $v0 for the return. Save the argument into
  // a virtual register so that we can access it from the return points.
  if (DAG.getMachineFunction().getFunction()->hasStructRetAttr()) {
    unsigned Reg = MipsFI->getSRetReturnReg();
    if (!Reg) {
      Reg = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i32));
      MipsFI->setSRetReturnReg(Reg);
    }
    SDValue Copy = DAG.getCopyToReg(DAG.getEntryNode(), Reg, ArgValues[0]);
    Root = DAG.getNode(ISD::TokenFactor, MVT::Other, Copy, Root);
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  return DAG.getMergeValues(Op.getNode()->getVTList(), &ArgValues[0],
                            ArgValues.size()).getValue(Op.getResNo());
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

SDValue MipsTargetLowering::
LowerRET(SDValue Op, SelectionDAG &DAG)
{
  // CCValAssign - represent the assignment of
  // the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;
  unsigned CC   = DAG.getMachineFunction().getFunction()->getCallingConv();
  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CC, isVarArg, getTargetMachine(), RVLocs);

  // Analize return values of ISD::RET
  CCInfo.AnalyzeReturn(Op.getNode(), RetCC_Mips);

  // If this is the first return lowered for this function, add 
  // the regs to the liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  // The chain is always operand #0
  SDValue Chain = Op.getOperand(0);
  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    // ISD::RET => ret chain, (regnum1,val1), ...
    // So i*2+1 index only the regnums
    Chain = DAG.getCopyToReg(Chain, VA.getLocReg(), Op.getOperand(i*2+1), Flag);

    // guarantee that all emitted copies are
    // stuck together, avoiding something bad
    Flag = Chain.getValue(1);
  }

  // The mips ABIs for returning structs by value requires that we copy
  // the sret argument into $v0 for the return. We saved the argument into
  // a virtual register in the entry block, so now we copy the value out
  // and into $v0.
  if (DAG.getMachineFunction().getFunction()->hasStructRetAttr()) {
    MachineFunction &MF      = DAG.getMachineFunction();
    MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();
    unsigned Reg = MipsFI->getSRetReturnReg();

    if (!Reg) 
      assert(0 && "sret virtual register not created in the entry block");
    SDValue Val = DAG.getCopyFromReg(Chain, Reg, getPointerTy());

    Chain = DAG.getCopyToReg(Chain, Mips::V0, Val, Flag);
    Flag = Chain.getValue(1);
  }

  // Return on Mips is always a "jr $ra"
  if (Flag.getNode())
    return DAG.getNode(MipsISD::Ret, MVT::Other, 
                       Chain, DAG.getRegister(Mips::RA, MVT::i32), Flag);
  else // Return Void
    return DAG.getNode(MipsISD::Ret, MVT::Other, 
                       Chain, DAG.getRegister(Mips::RA, MVT::i32));
}

//===----------------------------------------------------------------------===//
//                           Mips Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
MipsTargetLowering::ConstraintType MipsTargetLowering::
getConstraintType(const std::string &Constraint) const 
{
  // Mips specific constrainy 
  // GCC config/mips/constraints.md
  //
  // 'd' : An address register. Equivalent to r 
  //       unless generating MIPS16 code. 
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
        break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

/// getRegClassForInlineAsmConstraint - Given a constraint letter (e.g. "r"),
/// return a list of registers that can be used to satisfy the constraint.
/// This should only be used for C_RegisterClass constraints.
std::pair<unsigned, const TargetRegisterClass*> MipsTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint, MVT VT) const
{
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, Mips::CPURegsRegisterClass);
    case 'f':
      if (VT == MVT::f32) {
        if (Subtarget->isSingleFloat())
          return std::make_pair(0U, Mips::FGR32RegisterClass);
        else
          return std::make_pair(0U, Mips::AFGR32RegisterClass);
      }
      if (VT == MVT::f64)    
        if ((!Subtarget->isSingleFloat()) && (!Subtarget->isFP64bit()))
          return std::make_pair(0U, Mips::AFGR64RegisterClass);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

/// Given a register class constraint, like 'r', if this corresponds directly
/// to an LLVM register class, return a register of 0 and the register class
/// pointer.
std::vector<unsigned> MipsTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT VT) const
{
  if (Constraint.size() != 1)
    return std::vector<unsigned>();

  switch (Constraint[0]) {         
    default : break;
    case 'r':
    // GCC Mips Constraint Letters
    case 'd':     
    case 'y': 
      return make_vector<unsigned>(Mips::T0, Mips::T1, Mips::T2, Mips::T3, 
             Mips::T4, Mips::T5, Mips::T6, Mips::T7, Mips::S0, Mips::S1, 
             Mips::S2, Mips::S3, Mips::S4, Mips::S5, Mips::S6, Mips::S7, 
             Mips::T8, 0);

    case 'f':
      if (VT == MVT::f32) {
        if (Subtarget->isSingleFloat())
          return make_vector<unsigned>(Mips::F2, Mips::F3, Mips::F4, Mips::F5,
                 Mips::F6, Mips::F7, Mips::F8, Mips::F9, Mips::F10, Mips::F11,
                 Mips::F20, Mips::F21, Mips::F22, Mips::F23, Mips::F24,
                 Mips::F25, Mips::F26, Mips::F27, Mips::F28, Mips::F29,
                 Mips::F30, Mips::F31, 0);
        else
          return make_vector<unsigned>(Mips::F2, Mips::F4, Mips::F6, Mips::F8, 
                 Mips::F10, Mips::F20, Mips::F22, Mips::F24, Mips::F26, 
                 Mips::F28, Mips::F30, 0);
      }

      if (VT == MVT::f64)    
        if ((!Subtarget->isSingleFloat()) && (!Subtarget->isFP64bit()))
          return make_vector<unsigned>(Mips::D1, Mips::D2, Mips::D3, Mips::D4, 
                 Mips::D5, Mips::D10, Mips::D11, Mips::D12, Mips::D13, 
                 Mips::D14, Mips::D15, 0);
  }
  return std::vector<unsigned>();
}
