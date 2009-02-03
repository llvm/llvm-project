//===-- XCoreISelLowering.cpp - XCore DAG Lowering Implementation   ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the XCoreTargetLowering class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "xcore-lower"

#include "XCoreISelLowering.h"
#include "XCoreMachineFunctionInfo.h"
#include "XCore.h"
#include "XCoreTargetMachine.h"
#include "XCoreSubtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/VectorExtras.h"
#include <queue>
#include <set>
using namespace llvm;

const char *XCoreTargetLowering::
getTargetNodeName(unsigned Opcode) const 
{
  switch (Opcode) 
  {
    case XCoreISD::BL                : return "XCoreISD::BL";
    case XCoreISD::PCRelativeWrapper : return "XCoreISD::PCRelativeWrapper";
    case XCoreISD::DPRelativeWrapper : return "XCoreISD::DPRelativeWrapper";
    case XCoreISD::CPRelativeWrapper : return "XCoreISD::CPRelativeWrapper";
    case XCoreISD::STWSP             : return "XCoreISD::STWSP";
    case XCoreISD::RETSP             : return "XCoreISD::RETSP";
    default                           : return NULL;
  }
}

XCoreTargetLowering::XCoreTargetLowering(XCoreTargetMachine &XTM)
  : TargetLowering(XTM),
    TM(XTM),
    Subtarget(*XTM.getSubtargetImpl()) {

  // Set up the register classes.
  addRegisterClass(MVT::i32, XCore::GRRegsRegisterClass);

  // Compute derived properties from the register classes
  computeRegisterProperties();

  // Division is expensive
  setIntDivIsCheap(false);

  setShiftAmountType(MVT::i32);
  // shl X, 32 == 0
  setShiftAmountFlavor(Extend);
  setStackPointerRegisterToSaveRestore(XCore::SP);

  setSchedulingPreference(SchedulingForRegPressure);

  // Use i32 for setcc operations results (slt, sgt, ...).
  setBooleanContents(ZeroOrOneBooleanContent);

  // XCore does not have the NodeTypes below.
  setOperationAction(ISD::BR_CC,     MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32,   Custom);
  setOperationAction(ISD::ADDC, MVT::i32, Expand);
  setOperationAction(ISD::ADDE, MVT::i32, Expand);
  setOperationAction(ISD::SUBC, MVT::i32, Expand);
  setOperationAction(ISD::SUBE, MVT::i32, Expand);

  // Stop the combiner recombining select and set_cc
  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);
  
  // 64bit
  if (!Subtarget.isXS1A()) {
    setOperationAction(ISD::ADD, MVT::i64, Custom);
    setOperationAction(ISD::SUB, MVT::i64, Custom);
  }
  if (Subtarget.isXS1A()) {
    setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  }
  setOperationAction(ISD::MULHS, MVT::i32, Expand);
  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);
  
  // Bit Manipulation
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::ROTL , MVT::i32, Expand);
  setOperationAction(ISD::ROTR , MVT::i32, Expand);
  
  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  
  // Expand jump tables for now
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::JumpTable, MVT::i32, Custom);

  // RET must be custom lowered, to meet ABI requirements
  setOperationAction(ISD::RET,           MVT::Other, Custom);

  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  
  // Thread Local Storage
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  
  // Conversion of i64 -> double produces constantpool nodes
  setOperationAction(ISD::ConstantPool, MVT::i32,   Custom);

  // Loads
  setLoadExtAction(ISD::EXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);

  setLoadExtAction(ISD::SEXTLOAD, MVT::i8, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i16, Expand);
  
  // Varargs
  setOperationAction(ISD::VAEND, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  
  // Dynamic stack
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);
  
  // Debug
  setOperationAction(ISD::DBG_STOPPOINT, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
}

SDValue XCoreTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) 
  {
  case ISD::CALL:             return LowerCALL(Op, DAG);
  case ISD::FORMAL_ARGUMENTS: return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::RET:              return LowerRET(Op, DAG);
  case ISD::GlobalAddress:    return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress: return LowerGlobalTLSAddress(Op, DAG);
  case ISD::ConstantPool:     return LowerConstantPool(Op, DAG);
  case ISD::JumpTable:        return LowerJumpTable(Op, DAG);
  case ISD::SELECT_CC:        return LowerSELECT_CC(Op, DAG);
  case ISD::VAARG:            return LowerVAARG(Op, DAG);
  case ISD::VASTART:          return LowerVASTART(Op, DAG);
  // FIXME: Remove these when LegalizeDAGTypes lands.
  case ISD::ADD:
  case ISD::SUB:              return ExpandADDSUB(Op.getNode(), DAG);
  case ISD::FRAMEADDR:        return LowerFRAMEADDR(Op, DAG);
  default:
    assert(0 && "unimplemented operand");
    return SDValue();
  }
}

/// ReplaceNodeResults - Replace the results of node with an illegal result
/// type with new values built out of custom code.
void XCoreTargetLowering::ReplaceNodeResults(SDNode *N,
                                             SmallVectorImpl<SDValue>&Results,
                                             SelectionDAG &DAG) {
  switch (N->getOpcode()) {
  default:
    assert(0 && "Don't know how to custom expand this!");
    return;
  case ISD::ADD:
  case ISD::SUB:
    Results.push_back(ExpandADDSUB(N, DAG));
    return;
  }
}

//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//

SDValue XCoreTargetLowering::
LowerSELECT_CC(SDValue Op, SelectionDAG &DAG)
{
  SDValue Cond = DAG.getNode(ISD::SETCC, MVT::i32, Op.getOperand(2),
                             Op.getOperand(3), Op.getOperand(4));
  return DAG.getNode(ISD::SELECT, MVT::i32, Cond, Op.getOperand(0),
                     Op.getOperand(1));
}

SDValue XCoreTargetLowering::
getGlobalAddressWrapper(SDValue GA, GlobalValue *GV, SelectionDAG &DAG)
{
  if (isa<Function>(GV)) {
    return DAG.getNode(XCoreISD::PCRelativeWrapper, MVT::i32, GA);
  } else if (!Subtarget.isXS1A()) {
    const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
    if (!GVar) {
      // If GV is an alias then use the aliasee to determine constness
      if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
        GVar = dyn_cast_or_null<GlobalVariable>(GA->resolveAliasedGlobal());
    }
    bool isConst = GVar && GVar->isConstant();
    if (isConst) {
      return DAG.getNode(XCoreISD::CPRelativeWrapper, MVT::i32, GA);
    }
  }
  return DAG.getNode(XCoreISD::DPRelativeWrapper, MVT::i32, GA);
}

SDValue XCoreTargetLowering::
LowerGlobalAddress(SDValue Op, SelectionDAG &DAG)
{
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDValue GA = DAG.getTargetGlobalAddress(GV, MVT::i32);
  // If it's a debug information descriptor, don't mess with it.
  if (DAG.isVerifiedDebugInfoDesc(Op))
    return GA;
  return getGlobalAddressWrapper(GA, GV, DAG);
}

static inline SDValue BuildGetId(SelectionDAG &DAG) {
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, MVT::i32,
                     DAG.getConstant(Intrinsic::xcore_getid, MVT::i32));
}

static inline bool isZeroLengthArray(const Type *Ty) {
  const ArrayType *AT = dyn_cast_or_null<ArrayType>(Ty);
  return AT && (AT->getNumElements() == 0);
}

SDValue XCoreTargetLowering::
LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG)
{
  // transform to label + getid() * size
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDValue GA = DAG.getTargetGlobalAddress(GV, MVT::i32);
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (!GVar) {
    // If GV is an alias then use the aliasee to determine size
    if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
      GVar = dyn_cast_or_null<GlobalVariable>(GA->resolveAliasedGlobal());
  }
  if (! GVar) {
    assert(0 && "Thread local object not a GlobalVariable?");
    return SDValue();
  }
  const Type *Ty = cast<PointerType>(GV->getType())->getElementType();
  if (!Ty->isSized() || isZeroLengthArray(Ty)) {
    cerr << "Size of thread local object " << GVar->getName()
         << " is unknown\n";
    abort();
  }
  SDValue base = getGlobalAddressWrapper(GA, GV, DAG);
  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypePaddedSize(Ty);
  SDValue offset = DAG.getNode(ISD::MUL, MVT::i32, BuildGetId(DAG),
                       DAG.getConstant(Size, MVT::i32));
  return DAG.getNode(ISD::ADD, MVT::i32, base, offset);
}

SDValue XCoreTargetLowering::
LowerConstantPool(SDValue Op, SelectionDAG &DAG)
{
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  if (Subtarget.isXS1A()) {
    assert(0 && "Lowering of constant pool unimplemented");
    return SDValue();
  } else {
    MVT PtrVT = Op.getValueType();
    SDValue Res;
    if (CP->isMachineConstantPoolEntry()) {
      Res = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
                                      CP->getAlignment());
    } else {
      Res = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
                                      CP->getAlignment());
    }
    return DAG.getNode(XCoreISD::CPRelativeWrapper, MVT::i32, Res);
  }
}

SDValue XCoreTargetLowering::
LowerJumpTable(SDValue Op, SelectionDAG &DAG)
{
  MVT PtrVT = Op.getValueType();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDValue JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT);
  return DAG.getNode(XCoreISD::DPRelativeWrapper, MVT::i32, JTI);
}

SDValue XCoreTargetLowering::
ExpandADDSUB(SDNode *N, SelectionDAG &DAG)
{
  assert(N->getValueType(0) == MVT::i64 &&
         (N->getOpcode() == ISD::ADD || N->getOpcode() == ISD::SUB) &&
        "Unknown operand to lower!");
  assert(!Subtarget.isXS1A() && "Cannot custom lower ADD/SUB on xs1a");
  
  // Extract components
  SDValue LHSL = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, N->getOperand(0),
                             DAG.getConstant(0, MVT::i32));
  SDValue LHSH = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, N->getOperand(0),
                             DAG.getConstant(1, MVT::i32));
  SDValue RHSL = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, N->getOperand(1),
                             DAG.getConstant(0, MVT::i32));
  SDValue RHSH = DAG.getNode(ISD::EXTRACT_ELEMENT, MVT::i32, N->getOperand(1),
                             DAG.getConstant(1, MVT::i32));
  
  // Expand
  unsigned Opcode = (N->getOpcode() == ISD::ADD) ? XCoreISD::LADD :
                                                   XCoreISD::LSUB;
  SDValue Zero = DAG.getConstant(0, MVT::i32);
  SDValue Carry = DAG.getNode(Opcode, DAG.getVTList(MVT::i32, MVT::i32),
                                  LHSL, RHSL, Zero);
  SDValue Lo(Carry.getNode(), 1);
  
  SDValue Ignored = DAG.getNode(Opcode, DAG.getVTList(MVT::i32, MVT::i32),
                                  LHSH, RHSH, Carry);
  SDValue Hi(Ignored.getNode(), 1);
  // Merge the pieces
  return DAG.getNode(ISD::BUILD_PAIR, MVT::i64, Lo, Hi);
}

SDValue XCoreTargetLowering::
LowerVAARG(SDValue Op, SelectionDAG &DAG)
{
  assert(0 && "unimplemented");
  // FIX Arguments passed by reference need a extra dereference.
  SDNode *Node = Op.getNode();
  const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  MVT VT = Node->getValueType(0);
  SDValue VAList = DAG.getLoad(getPointerTy(), Node->getOperand(0),
                               Node->getOperand(1), V, 0);
  // Increment the pointer, VAList, to the next vararg
  SDValue Tmp3 = DAG.getNode(ISD::ADD, getPointerTy(), VAList, 
                     DAG.getConstant(VT.getSizeInBits(), 
                                     getPointerTy()));
  // Store the incremented VAList to the legalized pointer
  Tmp3 = DAG.getStore(VAList.getValue(1), Tmp3, Node->getOperand(1), V, 0);
  // Load the actual argument out of the pointer VAList
  return DAG.getLoad(VT, Tmp3, VAList, NULL, 0);
}

SDValue XCoreTargetLowering::
LowerVASTART(SDValue Op, SelectionDAG &DAG)
{
  // vastart stores the address of the VarArgsFrameIndex slot into the
  // memory location argument
  MachineFunction &MF = DAG.getMachineFunction();
  XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();
  SDValue Addr = DAG.getFrameIndex(XFI->getVarArgsFrameIndex(), MVT::i32);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), Addr, Op.getOperand(1), SV, 0);
}

SDValue XCoreTargetLowering::LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) {
  // Depths > 0 not supported yet! 
  if (cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() > 0)
    return SDValue();
  
  MachineFunction &MF = DAG.getMachineFunction();
  const TargetRegisterInfo *RegInfo = getTargetMachine().getRegisterInfo();
  return DAG.getCopyFromReg(DAG.getEntryNode(), RegInfo->getFrameRegister(MF),
    MVT::i32);
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

#include "XCoreGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//                  CALL Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// XCore custom CALL implementation
SDValue XCoreTargetLowering::
LowerCALL(SDValue Op, SelectionDAG &DAG)
{
  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  unsigned CallingConv = TheCall->getCallingConv();
  // For now, only CallingConv::C implemented
  switch (CallingConv) 
  {
    default:
      assert(0 && "Unsupported calling convention");
    case CallingConv::Fast:
    case CallingConv::C:
      return LowerCCCCallTo(Op, DAG, CallingConv);
  }
}

/// LowerCCCCallTo - functions arguments are copied from virtual
/// regs to (physical regs)/(stack frame), CALLSEQ_START and
/// CALLSEQ_END are emitted.
/// TODO: isTailCall, sret.
SDValue XCoreTargetLowering::
LowerCCCCallTo(SDValue Op, SelectionDAG &DAG, unsigned CC) 
{
  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  SDValue Chain  = TheCall->getChain();
  SDValue Callee = TheCall->getCallee();
  bool isVarArg  = TheCall->isVarArg();

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);

  // The ABI dictates there should be one stack slot available to the callee
  // on function entry (for saving lr).
  CCInfo.AllocateStack(4, 4);

  CCInfo.AnalyzeCallOperands(TheCall, CC_XCore);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(NumBytes, 
                                 getPointerTy(), true));

  SmallVector<std::pair<unsigned, SDValue>, 4> RegsToPass;
  SmallVector<SDValue, 12> MemOpChains;

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
    } else {
      assert(VA.isMemLoc());

      int Offset = VA.getLocMemOffset();

      MemOpChains.push_back(DAG.getNode(XCoreISD::STWSP, MVT::Other, Chain, Arg,
                                        DAG.getConstant(Offset/4, MVT::i32)));
    }
  }

  // Transform all store nodes into one single node because
  // all store nodes are independent of each other.
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

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  // XCoreBranchLink = #chain, #target_address, #opt_in_flags...
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

  Chain  = DAG.getNode(XCoreISD::BL, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, getPointerTy(), true),
                             DAG.getConstant(0, getPointerTy(), true),
                             InFlag);
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return SDValue(LowerCallResult(Chain, InFlag, TheCall, CC, DAG),
                 Op.getResNo());
}

/// LowerCallResult - Lower the result values of an ISD::CALL into the
/// appropriate copies out of appropriate physical registers.  This assumes that
/// Chain/InFlag are the input chain/flag to use, and that TheCall is the call
/// being lowered. Returns a SDNode with the same number of values as the 
/// ISD::CALL.
SDNode *XCoreTargetLowering::
LowerCallResult(SDValue Chain, SDValue InFlag, CallSDNode *TheCall, 
        unsigned CallingConv, SelectionDAG &DAG) {
  bool isVarArg = TheCall->isVarArg();

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallingConv, isVarArg, getTargetMachine(), RVLocs);

  CCInfo.AnalyzeCallResult(TheCall, RetCC_XCore);
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
  return DAG.getNode(ISD::MERGE_VALUES, TheCall->getVTList(),
                     &ResultVals[0], ResultVals.size()).getNode();
}

//===----------------------------------------------------------------------===//
//             FORMAL_ARGUMENTS Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// XCore custom FORMAL_ARGUMENTS implementation
SDValue XCoreTargetLowering::
LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG) 
{
  unsigned CC = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  switch(CC) 
  {
    default:
      assert(0 && "Unsupported calling convention");
    case CallingConv::C:
    case CallingConv::Fast:
      return LowerCCCArguments(Op, DAG);
  }
}

/// LowerCCCArguments - transform physical registers into
/// virtual registers and generate load operations for
/// arguments places on the stack.
/// TODO: sret
SDValue XCoreTargetLowering::
LowerCCCArguments(SDValue Op, SelectionDAG &DAG)
{
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  SDValue Root = Op.getOperand(0);
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue() != 0;
  unsigned CC = MF.getFunction()->getCallingConv();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);

  CCInfo.AnalyzeFormalArguments(Op.getNode(), CC_XCore);

  unsigned StackSlotSize = XCoreFrameInfo::stackSlotSize();

  SmallVector<SDValue, 16> ArgValues;
  
  unsigned LRSaveSize = StackSlotSize;
  
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {

    CCValAssign &VA = ArgLocs[i];
    
    if (VA.isRegLoc()) {
      // Arguments passed in registers
      MVT RegVT = VA.getLocVT();
      switch (RegVT.getSimpleVT()) {
      default:
        cerr << "LowerFORMAL_ARGUMENTS Unhandled argument type: "
             << RegVT.getSimpleVT()
             << "\n";
        abort();
      case MVT::i32:
        unsigned VReg = RegInfo.createVirtualRegister(
                          XCore::GRRegsRegisterClass);
        RegInfo.addLiveIn(VA.getLocReg(), VReg);
        ArgValues.push_back(DAG.getCopyFromReg(Root, VReg, RegVT));
      }
    } else {
      // sanity check
      assert(VA.isMemLoc());
      // Load the argument to a virtual register
      unsigned ObjSize = VA.getLocVT().getSizeInBits()/8;
      if (ObjSize > StackSlotSize) {
        cerr << "LowerFORMAL_ARGUMENTS Unhandled argument type: "
             << VA.getLocVT().getSimpleVT()
             << "\n";
      }
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(ObjSize,
                                      LRSaveSize + VA.getLocMemOffset());

      // Create the SelectionDAG nodes corresponding to a load
      //from this parameter
      SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
      ArgValues.push_back(DAG.getLoad(VA.getLocVT(), Root, FIN, NULL, 0));
    }
  }
  
  if (isVarArg) {
    /* Argument registers */
    static const unsigned ArgRegs[] = {
      XCore::R0, XCore::R1, XCore::R2, XCore::R3
    };
    XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();
    unsigned FirstVAReg = CCInfo.getFirstUnallocated(ArgRegs,
                                                     array_lengthof(ArgRegs));
    if (FirstVAReg < array_lengthof(ArgRegs)) {
      SmallVector<SDValue, 4> MemOps;
      int offset = 0;
      // Save remaining registers, storing higher register numbers at a higher
      // address
      for (unsigned i = array_lengthof(ArgRegs) - 1; i >= FirstVAReg; --i) {
        // Create a stack slot
        int FI = MFI->CreateFixedObject(4, offset);
        if (i == FirstVAReg) {
          XFI->setVarArgsFrameIndex(FI);
        }
        offset -= StackSlotSize;
        SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
        // Move argument from phys reg -> virt reg
        unsigned VReg = RegInfo.createVirtualRegister(
                          XCore::GRRegsRegisterClass);
        RegInfo.addLiveIn(ArgRegs[i], VReg);
        SDValue Val = DAG.getCopyFromReg(Root, VReg, MVT::i32);
        // Move argument from virt reg -> stack
        SDValue Store = DAG.getStore(Val.getValue(1), Val, FIN, NULL, 0);
        MemOps.push_back(Store);
      }
      if (!MemOps.empty())
        Root = DAG.getNode(ISD::TokenFactor, MVT::Other,
                           &MemOps[0], MemOps.size());
    } else {
      // This will point to the next argument passed via stack.
      XFI->setVarArgsFrameIndex(
          MFI->CreateFixedObject(4, LRSaveSize + CCInfo.getNextStackOffset()));
    }
  }
  
  ArgValues.push_back(Root);

  // Return the new list of results.
  std::vector<MVT> RetVT(Op.getNode()->value_begin(),
                                    Op.getNode()->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVT, &ArgValues[0], ArgValues.size());
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

SDValue XCoreTargetLowering::
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
  CCInfo.AnalyzeReturn(Op.getNode(), RetCC_XCore);

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

  // Return on XCore is always a "retsp 0"
  if (Flag.getNode())
    return DAG.getNode(XCoreISD::RETSP, MVT::Other,
                       Chain, DAG.getConstant(0, MVT::i32), Flag);
  else // Return Void
    return DAG.getNode(XCoreISD::RETSP, MVT::Other,
                       Chain, DAG.getConstant(0, MVT::i32));
}

//===----------------------------------------------------------------------===//
//  Other Lowering Code
//===----------------------------------------------------------------------===//

MachineBasicBlock *
XCoreTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                               MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *getTargetMachine().getInstrInfo();
  assert((MI->getOpcode() == XCore::SELECT_CC) &&
         "Unexpected instr type to insert");
  
  // To "insert" a SELECT_CC instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = BB;
  ++It;
  
  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   cmpTY ccX, r1, r2
  //   bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = F->CreateMachineBasicBlock(LLVM_BB);
  BuildMI(BB, TII.get(XCore::BRFT_lru6))
    .addReg(MI->getOperand(1).getReg()).addMBB(sinkMBB);
  F->insert(It, copy0MBB);
  F->insert(It, sinkMBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  sinkMBB->transferSuccessors(BB);
  // Next, add the true and fallthrough blocks as its successors.
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
  BuildMI(BB, TII.get(XCore::PHI), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(3).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);
  
  F->DeleteMachineInstr(MI);   // The pseudo instruction is gone now.
  return BB;
}

//===----------------------------------------------------------------------===//
//  Addressing mode description hooks
//===----------------------------------------------------------------------===//

static inline bool isImmUs(int64_t val)
{
  return (val >= 0 && val <= 11);
}

static inline bool isImmUs2(int64_t val)
{
  return (val%2 == 0 && isImmUs(val/2));
}

static inline bool isImmUs4(int64_t val)
{
  return (val%4 == 0 && isImmUs(val/4));
}

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool
XCoreTargetLowering::isLegalAddressingMode(const AddrMode &AM, 
                                              const Type *Ty) const {
  MVT VT = getValueType(Ty, true);
  // Get expected value type after legalization
  switch (VT.getSimpleVT()) {
  // Legal load / stores
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    break;
  // Expand i1 -> i8
  case MVT::i1:
    VT = MVT::i8;
    break;
  // Everything else is lowered to words
  default:
    VT = MVT::i32;
    break;
  }
  if (AM.BaseGV) {
    return VT == MVT::i32 && !AM.HasBaseReg && AM.Scale == 0 &&
                 AM.BaseOffs%4 == 0;
  }
  
  switch (VT.getSimpleVT()) {
  default:
    return false;
  case MVT::i8:
    // reg + imm
    if (AM.Scale == 0) {
      return isImmUs(AM.BaseOffs);
    }
    return AM.Scale == 1 && AM.BaseOffs == 0;
  case MVT::i16:
    // reg + imm
    if (AM.Scale == 0) {
      return isImmUs2(AM.BaseOffs);
    }
    return AM.Scale == 2 && AM.BaseOffs == 0;
  case MVT::i32:
    // reg + imm
    if (AM.Scale == 0) {
      return isImmUs4(AM.BaseOffs);
    }
    // reg + reg<<2
    return AM.Scale == 4 && AM.BaseOffs == 0;
  }
  
  return false;
}

//===----------------------------------------------------------------------===//
//                           XCore Inline Assembly Support
//===----------------------------------------------------------------------===//

std::vector<unsigned> XCoreTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT VT) const 
{
  if (Constraint.size() != 1)
    return std::vector<unsigned>();

  switch (Constraint[0]) {
    default : break;
    case 'r':
      return make_vector<unsigned>(XCore::R0, XCore::R1,  XCore::R2, 
                                   XCore::R3, XCore::R4,  XCore::R5, 
                                   XCore::R6, XCore::R7,  XCore::R8, 
                                   XCore::R9, XCore::R10, XCore::R11, 0);
      break;
  }
  return std::vector<unsigned>();
}
