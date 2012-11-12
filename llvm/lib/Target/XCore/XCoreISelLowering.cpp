//===-- XCoreISelLowering.cpp - XCore DAG Lowering Implementation ---------===//
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
#include "XCoreTargetObjectFile.h"
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
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
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
    case XCoreISD::LADD              : return "XCoreISD::LADD";
    case XCoreISD::LSUB              : return "XCoreISD::LSUB";
    case XCoreISD::LMUL              : return "XCoreISD::LMUL";
    case XCoreISD::MACCU             : return "XCoreISD::MACCU";
    case XCoreISD::MACCS             : return "XCoreISD::MACCS";
    case XCoreISD::BR_JT             : return "XCoreISD::BR_JT";
    case XCoreISD::BR_JT32           : return "XCoreISD::BR_JT32";
    default                          : return NULL;
  }
}

XCoreTargetLowering::XCoreTargetLowering(XCoreTargetMachine &XTM)
  : TargetLowering(XTM, new XCoreTargetObjectFile()),
    TM(XTM),
    Subtarget(*XTM.getSubtargetImpl()) {

  // Set up the register classes.
  addRegisterClass(MVT::i32, &XCore::GRRegsRegClass);

  // Compute derived properties from the register classes
  computeRegisterProperties();

  // Division is expensive
  setIntDivIsCheap(false);

  setStackPointerRegisterToSaveRestore(XCore::SP);

  setSchedulingPreference(Sched::RegPressure);

  // Use i32 for setcc operations results (slt, sgt, ...).
  setBooleanContents(ZeroOrOneBooleanContent);
  setBooleanVectorContents(ZeroOrOneBooleanContent); // FIXME: Is this correct?

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
  setOperationAction(ISD::ADD, MVT::i64, Custom);
  setOperationAction(ISD::SUB, MVT::i64, Custom);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Custom);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Custom);
  setOperationAction(ISD::MULHS, MVT::i32, Expand);
  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

  // Bit Manipulation
  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::ROTL , MVT::i32, Expand);
  setOperationAction(ISD::ROTR , MVT::i32, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);

  setOperationAction(ISD::TRAP, MVT::Other, Legal);

  // Jump tables.
  setOperationAction(ISD::BR_JT, MVT::Other, Custom);

  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  setOperationAction(ISD::BlockAddress, MVT::i32 , Custom);

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

  // Custom expand misaligned loads / stores.
  setOperationAction(ISD::LOAD, MVT::i32, Custom);
  setOperationAction(ISD::STORE, MVT::i32, Custom);

  // Varargs
  setOperationAction(ISD::VAEND, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  setOperationAction(ISD::VASTART, MVT::Other, Custom);

  // Dynamic stack
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32, Expand);

  // TRAMPOLINE is custom lowered.
  setOperationAction(ISD::INIT_TRAMPOLINE, MVT::Other, Custom);
  setOperationAction(ISD::ADJUST_TRAMPOLINE, MVT::Other, Custom);

  maxStoresPerMemset = maxStoresPerMemsetOptSize = 4;
  maxStoresPerMemmove = maxStoresPerMemmoveOptSize
    = maxStoresPerMemcpy = maxStoresPerMemcpyOptSize = 2;

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::ADD);

  setMinFunctionAlignment(1);
}

SDValue XCoreTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode())
  {
  case ISD::GlobalAddress:    return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress: return LowerGlobalTLSAddress(Op, DAG);
  case ISD::BlockAddress:     return LowerBlockAddress(Op, DAG);
  case ISD::ConstantPool:     return LowerConstantPool(Op, DAG);
  case ISD::BR_JT:            return LowerBR_JT(Op, DAG);
  case ISD::LOAD:             return LowerLOAD(Op, DAG);
  case ISD::STORE:            return LowerSTORE(Op, DAG);
  case ISD::SELECT_CC:        return LowerSELECT_CC(Op, DAG);
  case ISD::VAARG:            return LowerVAARG(Op, DAG);
  case ISD::VASTART:          return LowerVASTART(Op, DAG);
  case ISD::SMUL_LOHI:        return LowerSMUL_LOHI(Op, DAG);
  case ISD::UMUL_LOHI:        return LowerUMUL_LOHI(Op, DAG);
  // FIXME: Remove these when LegalizeDAGTypes lands.
  case ISD::ADD:
  case ISD::SUB:              return ExpandADDSUB(Op.getNode(), DAG);
  case ISD::FRAMEADDR:        return LowerFRAMEADDR(Op, DAG);
  case ISD::INIT_TRAMPOLINE:  return LowerINIT_TRAMPOLINE(Op, DAG);
  case ISD::ADJUST_TRAMPOLINE: return LowerADJUST_TRAMPOLINE(Op, DAG);
  default:
    llvm_unreachable("unimplemented operand");
  }
}

/// ReplaceNodeResults - Replace the results of node with an illegal result
/// type with new values built out of custom code.
void XCoreTargetLowering::ReplaceNodeResults(SDNode *N,
                                             SmallVectorImpl<SDValue>&Results,
                                             SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Don't know how to custom expand this!");
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
LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc dl = Op.getDebugLoc();
  SDValue Cond = DAG.getNode(ISD::SETCC, dl, MVT::i32, Op.getOperand(2),
                             Op.getOperand(3), Op.getOperand(4));
  return DAG.getNode(ISD::SELECT, dl, MVT::i32, Cond, Op.getOperand(0),
                     Op.getOperand(1));
}

SDValue XCoreTargetLowering::
getGlobalAddressWrapper(SDValue GA, const GlobalValue *GV,
                        SelectionDAG &DAG) const
{
  // FIXME there is no actual debug info here
  DebugLoc dl = GA.getDebugLoc();
  if (isa<Function>(GV)) {
    return DAG.getNode(XCoreISD::PCRelativeWrapper, dl, MVT::i32, GA);
  }
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (!GVar) {
    // If GV is an alias then use the aliasee to determine constness
    if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
      GVar = dyn_cast_or_null<GlobalVariable>(GA->resolveAliasedGlobal());
  }
  bool isConst = GVar && GVar->isConstant();
  if (isConst) {
    return DAG.getNode(XCoreISD::CPRelativeWrapper, dl, MVT::i32, GA);
  }
  return DAG.getNode(XCoreISD::DPRelativeWrapper, dl, MVT::i32, GA);
}

SDValue XCoreTargetLowering::
LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const
{
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDValue GA = DAG.getTargetGlobalAddress(GV, Op.getDebugLoc(), MVT::i32);
  return getGlobalAddressWrapper(GA, GV, DAG);
}

static inline SDValue BuildGetId(SelectionDAG &DAG, DebugLoc dl) {
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, MVT::i32,
                     DAG.getConstant(Intrinsic::xcore_getid, MVT::i32));
}

static inline bool isZeroLengthArray(Type *Ty) {
  ArrayType *AT = dyn_cast_or_null<ArrayType>(Ty);
  return AT && (AT->getNumElements() == 0);
}

SDValue XCoreTargetLowering::
LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const
{
  // FIXME there isn't really debug info here
  DebugLoc dl = Op.getDebugLoc();
  // transform to label + getid() * size
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDValue GA = DAG.getTargetGlobalAddress(GV, dl, MVT::i32);
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (!GVar) {
    // If GV is an alias then use the aliasee to determine size
    if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GV))
      GVar = dyn_cast_or_null<GlobalVariable>(GA->resolveAliasedGlobal());
  }
  if (!GVar) {
    llvm_unreachable("Thread local object not a GlobalVariable?");
  }
  Type *Ty = cast<PointerType>(GV->getType())->getElementType();
  if (!Ty->isSized() || isZeroLengthArray(Ty)) {
#ifndef NDEBUG
    errs() << "Size of thread local object " << GVar->getName()
           << " is unknown\n";
#endif
    llvm_unreachable(0);
  }
  SDValue base = getGlobalAddressWrapper(GA, GV, DAG);
  const DataLayout *TD = TM.getDataLayout();
  unsigned Size = TD->getTypeAllocSize(Ty);
  SDValue offset = DAG.getNode(ISD::MUL, dl, MVT::i32, BuildGetId(DAG, dl),
                       DAG.getConstant(Size, MVT::i32));
  return DAG.getNode(ISD::ADD, dl, MVT::i32, base, offset);
}

SDValue XCoreTargetLowering::
LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc DL = Op.getDebugLoc();

  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();
  SDValue Result = DAG.getTargetBlockAddress(BA, getPointerTy());

  return DAG.getNode(XCoreISD::PCRelativeWrapper, DL, getPointerTy(), Result);
}

SDValue XCoreTargetLowering::
LowerConstantPool(SDValue Op, SelectionDAG &DAG) const
{
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  // FIXME there isn't really debug info here
  DebugLoc dl = CP->getDebugLoc();
  EVT PtrVT = Op.getValueType();
  SDValue Res;
  if (CP->isMachineConstantPoolEntry()) {
    Res = DAG.getTargetConstantPool(CP->getMachineCPVal(), PtrVT,
                                    CP->getAlignment());
  } else {
    Res = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
                                    CP->getAlignment());
  }
  return DAG.getNode(XCoreISD::CPRelativeWrapper, dl, MVT::i32, Res);
}

unsigned XCoreTargetLowering::getJumpTableEncoding() const {
  return MachineJumpTableInfo::EK_Inline;
}

SDValue XCoreTargetLowering::
LowerBR_JT(SDValue Op, SelectionDAG &DAG) const
{
  SDValue Chain = Op.getOperand(0);
  SDValue Table = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);
  DebugLoc dl = Op.getDebugLoc();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Table);
  unsigned JTI = JT->getIndex();
  MachineFunction &MF = DAG.getMachineFunction();
  const MachineJumpTableInfo *MJTI = MF.getJumpTableInfo();
  SDValue TargetJT = DAG.getTargetJumpTable(JT->getIndex(), MVT::i32);

  unsigned NumEntries = MJTI->getJumpTables()[JTI].MBBs.size();
  if (NumEntries <= 32) {
    return DAG.getNode(XCoreISD::BR_JT, dl, MVT::Other, Chain, TargetJT, Index);
  }
  assert((NumEntries >> 31) == 0);
  SDValue ScaledIndex = DAG.getNode(ISD::SHL, dl, MVT::i32, Index,
                                    DAG.getConstant(1, MVT::i32));
  return DAG.getNode(XCoreISD::BR_JT32, dl, MVT::Other, Chain, TargetJT,
                     ScaledIndex);
}

static bool
IsWordAlignedBasePlusConstantOffset(SDValue Addr, SDValue &AlignedBase,
                                    int64_t &Offset)
{
  if (Addr.getOpcode() != ISD::ADD) {
    return false;
  }
  ConstantSDNode *CN = 0;
  if (!(CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))) {
    return false;
  }
  int64_t off = CN->getSExtValue();
  const SDValue &Base = Addr.getOperand(0);
  const SDValue *Root = &Base;
  if (Base.getOpcode() == ISD::ADD &&
      Base.getOperand(1).getOpcode() == ISD::SHL) {
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Base.getOperand(1)
                                                      .getOperand(1));
    if (CN && (CN->getSExtValue() >= 2)) {
      Root = &Base.getOperand(0);
    }
  }
  if (isa<FrameIndexSDNode>(*Root)) {
    // All frame indicies are word aligned
    AlignedBase = Base;
    Offset = off;
    return true;
  }
  if (Root->getOpcode() == XCoreISD::DPRelativeWrapper ||
      Root->getOpcode() == XCoreISD::CPRelativeWrapper) {
    // All dp / cp relative addresses are word aligned
    AlignedBase = Base;
    Offset = off;
    return true;
  }
  // Check for an aligned global variable.
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(*Root)) {
    const GlobalValue *GV = GA->getGlobal();
    if (GA->getOffset() == 0 && GV->getAlignment() >= 4) {
      AlignedBase = Base;
      Offset = off;
      return true;
    }
  }
  return false;
}

SDValue XCoreTargetLowering::
LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  LoadSDNode *LD = cast<LoadSDNode>(Op);
  assert(LD->getExtensionType() == ISD::NON_EXTLOAD &&
         "Unexpected extension type");
  assert(LD->getMemoryVT() == MVT::i32 && "Unexpected load EVT");
  if (allowsUnalignedMemoryAccesses(LD->getMemoryVT()))
    return SDValue();

  unsigned ABIAlignment = getDataLayout()->
    getABITypeAlignment(LD->getMemoryVT().getTypeForEVT(*DAG.getContext()));
  // Leave aligned load alone.
  if (LD->getAlignment() >= ABIAlignment)
    return SDValue();

  SDValue Chain = LD->getChain();
  SDValue BasePtr = LD->getBasePtr();
  DebugLoc DL = Op.getDebugLoc();

  SDValue Base;
  int64_t Offset;
  if (!LD->isVolatile() &&
      IsWordAlignedBasePlusConstantOffset(BasePtr, Base, Offset)) {
    if (Offset % 4 == 0) {
      // We've managed to infer better alignment information than the load
      // already has. Use an aligned load.
      //
      return DAG.getLoad(getPointerTy(), DL, Chain, BasePtr,
                         MachinePointerInfo(),
                         false, false, false, 0);
    }
    // Lower to
    // ldw low, base[offset >> 2]
    // ldw high, base[(offset >> 2) + 1]
    // shr low_shifted, low, (offset & 0x3) * 8
    // shl high_shifted, high, 32 - (offset & 0x3) * 8
    // or result, low_shifted, high_shifted
    SDValue LowOffset = DAG.getConstant(Offset & ~0x3, MVT::i32);
    SDValue HighOffset = DAG.getConstant((Offset & ~0x3) + 4, MVT::i32);
    SDValue LowShift = DAG.getConstant((Offset & 0x3) * 8, MVT::i32);
    SDValue HighShift = DAG.getConstant(32 - (Offset & 0x3) * 8, MVT::i32);

    SDValue LowAddr = DAG.getNode(ISD::ADD, DL, MVT::i32, Base, LowOffset);
    SDValue HighAddr = DAG.getNode(ISD::ADD, DL, MVT::i32, Base, HighOffset);

    SDValue Low = DAG.getLoad(getPointerTy(), DL, Chain,
                              LowAddr, MachinePointerInfo(),
                              false, false, false, 0);
    SDValue High = DAG.getLoad(getPointerTy(), DL, Chain,
                               HighAddr, MachinePointerInfo(),
                               false, false, false, 0);
    SDValue LowShifted = DAG.getNode(ISD::SRL, DL, MVT::i32, Low, LowShift);
    SDValue HighShifted = DAG.getNode(ISD::SHL, DL, MVT::i32, High, HighShift);
    SDValue Result = DAG.getNode(ISD::OR, DL, MVT::i32, LowShifted, HighShifted);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Low.getValue(1),
                             High.getValue(1));
    SDValue Ops[] = { Result, Chain };
    return DAG.getMergeValues(Ops, 2, DL);
  }

  if (LD->getAlignment() == 2) {
    SDValue Low = DAG.getExtLoad(ISD::ZEXTLOAD, DL, MVT::i32, Chain,
                                 BasePtr, LD->getPointerInfo(), MVT::i16,
                                 LD->isVolatile(), LD->isNonTemporal(), 2);
    SDValue HighAddr = DAG.getNode(ISD::ADD, DL, MVT::i32, BasePtr,
                                   DAG.getConstant(2, MVT::i32));
    SDValue High = DAG.getExtLoad(ISD::EXTLOAD, DL, MVT::i32, Chain,
                                  HighAddr,
                                  LD->getPointerInfo().getWithOffset(2),
                                  MVT::i16, LD->isVolatile(),
                                  LD->isNonTemporal(), 2);
    SDValue HighShifted = DAG.getNode(ISD::SHL, DL, MVT::i32, High,
                                      DAG.getConstant(16, MVT::i32));
    SDValue Result = DAG.getNode(ISD::OR, DL, MVT::i32, Low, HighShifted);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Low.getValue(1),
                             High.getValue(1));
    SDValue Ops[] = { Result, Chain };
    return DAG.getMergeValues(Ops, 2, DL);
  }

  // Lower to a call to __misaligned_load(BasePtr).
  Type *IntPtrTy = getDataLayout()->getIntPtrType(*DAG.getContext());
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;

  Entry.Ty = IntPtrTy;
  Entry.Node = BasePtr;
  Args.push_back(Entry);

  TargetLowering::CallLoweringInfo CLI(Chain, IntPtrTy, false, false,
                    false, false, 0, CallingConv::C, /*isTailCall=*/false,
                    /*doesNotRet=*/false, /*isReturnValueUsed=*/true,
                    DAG.getExternalSymbol("__misaligned_load", getPointerTy()),
                    Args, DAG, DL);
  std::pair<SDValue, SDValue> CallResult = LowerCallTo(CLI);

  SDValue Ops[] =
    { CallResult.first, CallResult.second };

  return DAG.getMergeValues(Ops, 2, DL);
}

SDValue XCoreTargetLowering::
LowerSTORE(SDValue Op, SelectionDAG &DAG) const
{
  StoreSDNode *ST = cast<StoreSDNode>(Op);
  assert(!ST->isTruncatingStore() && "Unexpected store type");
  assert(ST->getMemoryVT() == MVT::i32 && "Unexpected store EVT");
  if (allowsUnalignedMemoryAccesses(ST->getMemoryVT())) {
    return SDValue();
  }
  unsigned ABIAlignment = getDataLayout()->
    getABITypeAlignment(ST->getMemoryVT().getTypeForEVT(*DAG.getContext()));
  // Leave aligned store alone.
  if (ST->getAlignment() >= ABIAlignment) {
    return SDValue();
  }
  SDValue Chain = ST->getChain();
  SDValue BasePtr = ST->getBasePtr();
  SDValue Value = ST->getValue();
  DebugLoc dl = Op.getDebugLoc();

  if (ST->getAlignment() == 2) {
    SDValue Low = Value;
    SDValue High = DAG.getNode(ISD::SRL, dl, MVT::i32, Value,
                                      DAG.getConstant(16, MVT::i32));
    SDValue StoreLow = DAG.getTruncStore(Chain, dl, Low, BasePtr,
                                         ST->getPointerInfo(), MVT::i16,
                                         ST->isVolatile(), ST->isNonTemporal(),
                                         2);
    SDValue HighAddr = DAG.getNode(ISD::ADD, dl, MVT::i32, BasePtr,
                                   DAG.getConstant(2, MVT::i32));
    SDValue StoreHigh = DAG.getTruncStore(Chain, dl, High, HighAddr,
                                          ST->getPointerInfo().getWithOffset(2),
                                          MVT::i16, ST->isVolatile(),
                                          ST->isNonTemporal(), 2);
    return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, StoreLow, StoreHigh);
  }

  // Lower to a call to __misaligned_store(BasePtr, Value).
  Type *IntPtrTy = getDataLayout()->getIntPtrType(*DAG.getContext());
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;

  Entry.Ty = IntPtrTy;
  Entry.Node = BasePtr;
  Args.push_back(Entry);

  Entry.Node = Value;
  Args.push_back(Entry);

  TargetLowering::CallLoweringInfo CLI(Chain,
                    Type::getVoidTy(*DAG.getContext()), false, false,
                    false, false, 0, CallingConv::C, /*isTailCall=*/false,
                    /*doesNotRet=*/false, /*isReturnValueUsed=*/true,
                    DAG.getExternalSymbol("__misaligned_store", getPointerTy()),
                    Args, DAG, dl);
  std::pair<SDValue, SDValue> CallResult = LowerCallTo(CLI);

  return CallResult.second;
}

SDValue XCoreTargetLowering::
LowerSMUL_LOHI(SDValue Op, SelectionDAG &DAG) const
{
  assert(Op.getValueType() == MVT::i32 && Op.getOpcode() == ISD::SMUL_LOHI &&
         "Unexpected operand to lower!");
  DebugLoc dl = Op.getDebugLoc();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue Zero = DAG.getConstant(0, MVT::i32);
  SDValue Hi = DAG.getNode(XCoreISD::MACCS, dl,
                           DAG.getVTList(MVT::i32, MVT::i32), Zero, Zero,
                           LHS, RHS);
  SDValue Lo(Hi.getNode(), 1);
  SDValue Ops[] = { Lo, Hi };
  return DAG.getMergeValues(Ops, 2, dl);
}

SDValue XCoreTargetLowering::
LowerUMUL_LOHI(SDValue Op, SelectionDAG &DAG) const
{
  assert(Op.getValueType() == MVT::i32 && Op.getOpcode() == ISD::UMUL_LOHI &&
         "Unexpected operand to lower!");
  DebugLoc dl = Op.getDebugLoc();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue Zero = DAG.getConstant(0, MVT::i32);
  SDValue Hi = DAG.getNode(XCoreISD::LMUL, dl,
                           DAG.getVTList(MVT::i32, MVT::i32), LHS, RHS,
                           Zero, Zero);
  SDValue Lo(Hi.getNode(), 1);
  SDValue Ops[] = { Lo, Hi };
  return DAG.getMergeValues(Ops, 2, dl);
}

/// isADDADDMUL - Return whether Op is in a form that is equivalent to
/// add(add(mul(x,y),a),b). If requireIntermediatesHaveOneUse is true then
/// each intermediate result in the calculation must also have a single use.
/// If the Op is in the correct form the constituent parts are written to Mul0,
/// Mul1, Addend0 and Addend1.
static bool
isADDADDMUL(SDValue Op, SDValue &Mul0, SDValue &Mul1, SDValue &Addend0,
            SDValue &Addend1, bool requireIntermediatesHaveOneUse)
{
  if (Op.getOpcode() != ISD::ADD)
    return false;
  SDValue N0 = Op.getOperand(0);
  SDValue N1 = Op.getOperand(1);
  SDValue AddOp;
  SDValue OtherOp;
  if (N0.getOpcode() == ISD::ADD) {
    AddOp = N0;
    OtherOp = N1;
  } else if (N1.getOpcode() == ISD::ADD) {
    AddOp = N1;
    OtherOp = N0;
  } else {
    return false;
  }
  if (requireIntermediatesHaveOneUse && !AddOp.hasOneUse())
    return false;
  if (OtherOp.getOpcode() == ISD::MUL) {
    // add(add(a,b),mul(x,y))
    if (requireIntermediatesHaveOneUse && !OtherOp.hasOneUse())
      return false;
    Mul0 = OtherOp.getOperand(0);
    Mul1 = OtherOp.getOperand(1);
    Addend0 = AddOp.getOperand(0);
    Addend1 = AddOp.getOperand(1);
    return true;
  }
  if (AddOp.getOperand(0).getOpcode() == ISD::MUL) {
    // add(add(mul(x,y),a),b)
    if (requireIntermediatesHaveOneUse && !AddOp.getOperand(0).hasOneUse())
      return false;
    Mul0 = AddOp.getOperand(0).getOperand(0);
    Mul1 = AddOp.getOperand(0).getOperand(1);
    Addend0 = AddOp.getOperand(1);
    Addend1 = OtherOp;
    return true;
  }
  if (AddOp.getOperand(1).getOpcode() == ISD::MUL) {
    // add(add(a,mul(x,y)),b)
    if (requireIntermediatesHaveOneUse && !AddOp.getOperand(1).hasOneUse())
      return false;
    Mul0 = AddOp.getOperand(1).getOperand(0);
    Mul1 = AddOp.getOperand(1).getOperand(1);
    Addend0 = AddOp.getOperand(0);
    Addend1 = OtherOp;
    return true;
  }
  return false;
}

SDValue XCoreTargetLowering::
TryExpandADDWithMul(SDNode *N, SelectionDAG &DAG) const
{
  SDValue Mul;
  SDValue Other;
  if (N->getOperand(0).getOpcode() == ISD::MUL) {
    Mul = N->getOperand(0);
    Other = N->getOperand(1);
  } else if (N->getOperand(1).getOpcode() == ISD::MUL) {
    Mul = N->getOperand(1);
    Other = N->getOperand(0);
  } else {
    return SDValue();
  }
  DebugLoc dl = N->getDebugLoc();
  SDValue LL, RL, AddendL, AddendH;
  LL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                   Mul.getOperand(0),  DAG.getConstant(0, MVT::i32));
  RL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                   Mul.getOperand(1),  DAG.getConstant(0, MVT::i32));
  AddendL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                        Other,  DAG.getConstant(0, MVT::i32));
  AddendH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                        Other,  DAG.getConstant(1, MVT::i32));
  APInt HighMask = APInt::getHighBitsSet(64, 32);
  unsigned LHSSB = DAG.ComputeNumSignBits(Mul.getOperand(0));
  unsigned RHSSB = DAG.ComputeNumSignBits(Mul.getOperand(1));
  if (DAG.MaskedValueIsZero(Mul.getOperand(0), HighMask) &&
      DAG.MaskedValueIsZero(Mul.getOperand(1), HighMask)) {
    // The inputs are both zero-extended.
    SDValue Hi = DAG.getNode(XCoreISD::MACCU, dl,
                             DAG.getVTList(MVT::i32, MVT::i32), AddendH,
                             AddendL, LL, RL);
    SDValue Lo(Hi.getNode(), 1);
    return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Lo, Hi);
  }
  if (LHSSB > 32 && RHSSB > 32) {
    // The inputs are both sign-extended.
    SDValue Hi = DAG.getNode(XCoreISD::MACCS, dl,
                             DAG.getVTList(MVT::i32, MVT::i32), AddendH,
                             AddendL, LL, RL);
    SDValue Lo(Hi.getNode(), 1);
    return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Lo, Hi);
  }
  SDValue LH, RH;
  LH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                   Mul.getOperand(0),  DAG.getConstant(1, MVT::i32));
  RH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                   Mul.getOperand(1),  DAG.getConstant(1, MVT::i32));
  SDValue Hi = DAG.getNode(XCoreISD::MACCU, dl,
                           DAG.getVTList(MVT::i32, MVT::i32), AddendH,
                           AddendL, LL, RL);
  SDValue Lo(Hi.getNode(), 1);
  RH = DAG.getNode(ISD::MUL, dl, MVT::i32, LL, RH);
  LH = DAG.getNode(ISD::MUL, dl, MVT::i32, LH, RL);
  Hi = DAG.getNode(ISD::ADD, dl, MVT::i32, Hi, RH);
  Hi = DAG.getNode(ISD::ADD, dl, MVT::i32, Hi, LH);
  return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Lo, Hi);
}

SDValue XCoreTargetLowering::
ExpandADDSUB(SDNode *N, SelectionDAG &DAG) const
{
  assert(N->getValueType(0) == MVT::i64 &&
         (N->getOpcode() == ISD::ADD || N->getOpcode() == ISD::SUB) &&
        "Unknown operand to lower!");

  if (N->getOpcode() == ISD::ADD) {
    SDValue Result = TryExpandADDWithMul(N, DAG);
    if (Result.getNode() != 0)
      return Result;
  }

  DebugLoc dl = N->getDebugLoc();

  // Extract components
  SDValue LHSL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                            N->getOperand(0),  DAG.getConstant(0, MVT::i32));
  SDValue LHSH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                            N->getOperand(0),  DAG.getConstant(1, MVT::i32));
  SDValue RHSL = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                             N->getOperand(1), DAG.getConstant(0, MVT::i32));
  SDValue RHSH = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                             N->getOperand(1), DAG.getConstant(1, MVT::i32));

  // Expand
  unsigned Opcode = (N->getOpcode() == ISD::ADD) ? XCoreISD::LADD :
                                                   XCoreISD::LSUB;
  SDValue Zero = DAG.getConstant(0, MVT::i32);
  SDValue Carry = DAG.getNode(Opcode, dl, DAG.getVTList(MVT::i32, MVT::i32),
                                  LHSL, RHSL, Zero);
  SDValue Lo(Carry.getNode(), 1);

  SDValue Ignored = DAG.getNode(Opcode, dl, DAG.getVTList(MVT::i32, MVT::i32),
                                  LHSH, RHSH, Carry);
  SDValue Hi(Ignored.getNode(), 1);
  // Merge the pieces
  return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Lo, Hi);
}

SDValue XCoreTargetLowering::
LowerVAARG(SDValue Op, SelectionDAG &DAG) const
{
  llvm_unreachable("unimplemented");
  // FIXME Arguments passed by reference need a extra dereference.
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  const Value *V = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  EVT VT = Node->getValueType(0);
  SDValue VAList = DAG.getLoad(getPointerTy(), dl, Node->getOperand(0),
                               Node->getOperand(1), MachinePointerInfo(V),
                               false, false, false, 0);
  // Increment the pointer, VAList, to the next vararg
  SDValue Tmp3 = DAG.getNode(ISD::ADD, dl, getPointerTy(), VAList,
                     DAG.getConstant(VT.getSizeInBits(),
                                     getPointerTy()));
  // Store the incremented VAList to the legalized pointer
  Tmp3 = DAG.getStore(VAList.getValue(1), dl, Tmp3, Node->getOperand(1),
                      MachinePointerInfo(V), false, false, 0);
  // Load the actual argument out of the pointer VAList
  return DAG.getLoad(VT, dl, Tmp3, VAList, MachinePointerInfo(),
                     false, false, false, 0);
}

SDValue XCoreTargetLowering::
LowerVASTART(SDValue Op, SelectionDAG &DAG) const
{
  DebugLoc dl = Op.getDebugLoc();
  // vastart stores the address of the VarArgsFrameIndex slot into the
  // memory location argument
  MachineFunction &MF = DAG.getMachineFunction();
  XCoreFunctionInfo *XFI = MF.getInfo<XCoreFunctionInfo>();
  SDValue Addr = DAG.getFrameIndex(XFI->getVarArgsFrameIndex(), MVT::i32);
  return DAG.getStore(Op.getOperand(0), dl, Addr, Op.getOperand(1),
                      MachinePointerInfo(), false, false, 0);
}

SDValue XCoreTargetLowering::LowerFRAMEADDR(SDValue Op,
                                            SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  // Depths > 0 not supported yet!
  if (cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue() > 0)
    return SDValue();

  MachineFunction &MF = DAG.getMachineFunction();
  const TargetRegisterInfo *RegInfo = getTargetMachine().getRegisterInfo();
  return DAG.getCopyFromReg(DAG.getEntryNode(), dl,
                            RegInfo->getFrameRegister(MF), MVT::i32);
}

SDValue XCoreTargetLowering::
LowerADJUST_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const {
  return Op.getOperand(0);
}

SDValue XCoreTargetLowering::
LowerINIT_TRAMPOLINE(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Trmp = Op.getOperand(1); // trampoline
  SDValue FPtr = Op.getOperand(2); // nested function
  SDValue Nest = Op.getOperand(3); // 'nest' parameter value

  const Value *TrmpAddr = cast<SrcValueSDNode>(Op.getOperand(4))->getValue();

  // .align 4
  // LDAPF_u10 r11, nest
  // LDW_2rus r11, r11[0]
  // STWSP_ru6 r11, sp[0]
  // LDAPF_u10 r11, fptr
  // LDW_2rus r11, r11[0]
  // BAU_1r r11
  // nest:
  // .word nest
  // fptr:
  // .word fptr
  SDValue OutChains[5];

  SDValue Addr = Trmp;

  DebugLoc dl = Op.getDebugLoc();
  OutChains[0] = DAG.getStore(Chain, dl, DAG.getConstant(0x0a3cd805, MVT::i32),
                              Addr, MachinePointerInfo(TrmpAddr), false, false,
                              0);

  Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                     DAG.getConstant(4, MVT::i32));
  OutChains[1] = DAG.getStore(Chain, dl, DAG.getConstant(0xd80456c0, MVT::i32),
                              Addr, MachinePointerInfo(TrmpAddr, 4), false,
                              false, 0);

  Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                     DAG.getConstant(8, MVT::i32));
  OutChains[2] = DAG.getStore(Chain, dl, DAG.getConstant(0x27fb0a3c, MVT::i32),
                              Addr, MachinePointerInfo(TrmpAddr, 8), false,
                              false, 0);

  Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                     DAG.getConstant(12, MVT::i32));
  OutChains[3] = DAG.getStore(Chain, dl, Nest, Addr,
                              MachinePointerInfo(TrmpAddr, 12), false, false,
                              0);

  Addr = DAG.getNode(ISD::ADD, dl, MVT::i32, Trmp,
                     DAG.getConstant(16, MVT::i32));
  OutChains[4] = DAG.getStore(Chain, dl, FPtr, Addr,
                              MachinePointerInfo(TrmpAddr, 16), false, false,
                              0);

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains, 5);
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "XCoreGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//                  Call Calling Convention Implementation
//===----------------------------------------------------------------------===//

/// XCore call implementation
SDValue
XCoreTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
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

  // XCore target does not yet support tail call optimization.
  isTailCall = false;

  // For now, only CallingConv::C implemented
  switch (CallConv)
  {
    default:
      llvm_unreachable("Unsupported calling convention");
    case CallingConv::Fast:
    case CallingConv::C:
      return LowerCCCCallTo(Chain, Callee, CallConv, isVarArg, isTailCall,
                            Outs, OutVals, Ins, dl, DAG, InVals);
  }
}

/// LowerCCCCallTo - functions arguments are copied from virtual
/// regs to (physical regs)/(stack frame), CALLSEQ_START and
/// CALLSEQ_END are emitted.
/// TODO: isTailCall, sret.
SDValue
XCoreTargetLowering::LowerCCCCallTo(SDValue Chain, SDValue Callee,
                                    CallingConv::ID CallConv, bool isVarArg,
                                    bool isTailCall,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    DebugLoc dl, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &InVals) const {

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  // The ABI dictates there should be one stack slot available to the callee
  // on function entry (for saving lr).
  CCInfo.AllocateStack(4, 4);

  CCInfo.AnalyzeCallOperands(Outs, CC_XCore);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  Chain = DAG.getCALLSEQ_START(Chain,DAG.getConstant(NumBytes,
                                 getPointerTy(), true));

  SmallVector<std::pair<unsigned, SDValue>, 4> RegsToPass;
  SmallVector<SDValue, 12> MemOpChains;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
      default: llvm_unreachable("Unknown loc info!");
      case CCValAssign::Full: break;
      case CCValAssign::SExt:
        Arg = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), Arg);
        break;
      case CCValAssign::ZExt:
        Arg = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), Arg);
        break;
      case CCValAssign::AExt:
        Arg = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), Arg);
        break;
    }

    // Arguments that can be passed on register must be kept at
    // RegsToPass vector
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());

      int Offset = VA.getLocMemOffset();

      MemOpChains.push_back(DAG.getNode(XCoreISD::STWSP, dl, MVT::Other,
                                        Chain, Arg,
                                        DAG.getConstant(Offset/4, MVT::i32)));
    }
  }

  // Transform all store nodes into one single node because
  // all store nodes are independent of each other.
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

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl, MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  // XCoreBranchLink = #chain, #target_address, #opt_in_flags...
  //             = Chain, Callee, Reg#1, Reg#2, ...
  //
  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
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

  Chain  = DAG.getNode(XCoreISD::BL, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, getPointerTy(), true),
                             DAG.getConstant(0, getPointerTy(), true),
                             InFlag);
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg,
                         Ins, dl, DAG, InVals);
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
SDValue
XCoreTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                     CallingConv::ID CallConv, bool isVarArg,
                                     const SmallVectorImpl<ISD::InputArg> &Ins,
                                     DebugLoc dl, SelectionDAG &DAG,
                                     SmallVectorImpl<SDValue> &InVals) const {

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  CCInfo.AnalyzeCallResult(Ins, RetCC_XCore);

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

/// XCore formal arguments implementation
SDValue
XCoreTargetLowering::LowerFormalArguments(SDValue Chain,
                                          CallingConv::ID CallConv,
                                          bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                          DebugLoc dl,
                                          SelectionDAG &DAG,
                                          SmallVectorImpl<SDValue> &InVals)
                                            const {
  switch (CallConv)
  {
    default:
      llvm_unreachable("Unsupported calling convention");
    case CallingConv::C:
    case CallingConv::Fast:
      return LowerCCCArguments(Chain, CallConv, isVarArg,
                               Ins, dl, DAG, InVals);
  }
}

/// LowerCCCArguments - transform physical registers into
/// virtual registers and generate load operations for
/// arguments places on the stack.
/// TODO: sret
SDValue
XCoreTargetLowering::LowerCCCArguments(SDValue Chain,
                                       CallingConv::ID CallConv,
                                       bool isVarArg,
                                       const SmallVectorImpl<ISD::InputArg>
                                         &Ins,
                                       DebugLoc dl,
                                       SelectionDAG &DAG,
                                       SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  CCInfo.AnalyzeFormalArguments(Ins, CC_XCore);

  unsigned StackSlotSize = XCoreFrameLowering::stackSlotSize();

  unsigned LRSaveSize = StackSlotSize;

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {

    CCValAssign &VA = ArgLocs[i];

    if (VA.isRegLoc()) {
      // Arguments passed in registers
      EVT RegVT = VA.getLocVT();
      switch (RegVT.getSimpleVT().SimpleTy) {
      default:
        {
#ifndef NDEBUG
          errs() << "LowerFormalArguments Unhandled argument type: "
                 << RegVT.getSimpleVT().SimpleTy << "\n";
#endif
          llvm_unreachable(0);
        }
      case MVT::i32:
        unsigned VReg = RegInfo.createVirtualRegister(&XCore::GRRegsRegClass);
        RegInfo.addLiveIn(VA.getLocReg(), VReg);
        InVals.push_back(DAG.getCopyFromReg(Chain, dl, VReg, RegVT));
      }
    } else {
      // sanity check
      assert(VA.isMemLoc());
      // Load the argument to a virtual register
      unsigned ObjSize = VA.getLocVT().getSizeInBits()/8;
      if (ObjSize > StackSlotSize) {
        errs() << "LowerFormalArguments Unhandled argument type: "
               << EVT(VA.getLocVT()).getEVTString()
               << "\n";
      }
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(ObjSize,
                                      LRSaveSize + VA.getLocMemOffset(),
                                      true);

      // Create the SelectionDAG nodes corresponding to a load
      //from this parameter
      SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
      InVals.push_back(DAG.getLoad(VA.getLocVT(), dl, Chain, FIN,
                                   MachinePointerInfo::getFixedStack(FI),
                                   false, false, false, 0));
    }
  }

  if (isVarArg) {
    /* Argument registers */
    static const uint16_t ArgRegs[] = {
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
      for (int i = array_lengthof(ArgRegs) - 1; i >= (int)FirstVAReg; --i) {
        // Create a stack slot
        int FI = MFI->CreateFixedObject(4, offset, true);
        if (i == (int)FirstVAReg) {
          XFI->setVarArgsFrameIndex(FI);
        }
        offset -= StackSlotSize;
        SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
        // Move argument from phys reg -> virt reg
        unsigned VReg = RegInfo.createVirtualRegister(&XCore::GRRegsRegClass);
        RegInfo.addLiveIn(ArgRegs[i], VReg);
        SDValue Val = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i32);
        // Move argument from virt reg -> stack
        SDValue Store = DAG.getStore(Val.getValue(1), dl, Val, FIN,
                                     MachinePointerInfo(), false, false, 0);
        MemOps.push_back(Store);
      }
      if (!MemOps.empty())
        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                            &MemOps[0], MemOps.size());
    } else {
      // This will point to the next argument passed via stack.
      XFI->setVarArgsFrameIndex(
        MFI->CreateFixedObject(4, LRSaveSize + CCInfo.getNextStackOffset(),
                               true));
    }
  }

  return Chain;
}

//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

bool XCoreTargetLowering::
CanLowerReturn(CallingConv::ID CallConv, MachineFunction &MF,
               bool isVarArg,
               const SmallVectorImpl<ISD::OutputArg> &Outs,
               LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(), RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_XCore);
}

SDValue
XCoreTargetLowering::LowerReturn(SDValue Chain,
                                 CallingConv::ID CallConv, bool isVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 DebugLoc dl, SelectionDAG &DAG) const {

  // CCValAssign - represent the assignment of
  // the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  // Analyze return values.
  CCInfo.AnalyzeReturn(Outs, RetCC_XCore);

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

  // Return on XCore is always a "retsp 0"
  if (Flag.getNode())
    return DAG.getNode(XCoreISD::RETSP, dl, MVT::Other,
                       Chain, DAG.getConstant(0, MVT::i32), Flag);
  else // Return Void
    return DAG.getNode(XCoreISD::RETSP, dl, MVT::Other,
                       Chain, DAG.getConstant(0, MVT::i32));
}

//===----------------------------------------------------------------------===//
//  Other Lowering Code
//===----------------------------------------------------------------------===//

MachineBasicBlock *
XCoreTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                 MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
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
  F->insert(It, copy0MBB);
  F->insert(It, sinkMBB);

  // Transfer the remainder of BB and its successor edges to sinkMBB.
  sinkMBB->splice(sinkMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  sinkMBB->transferSuccessorsAndUpdatePHIs(BB);

  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  BuildMI(BB, dl, TII.get(XCore::BRFT_lru6))
    .addReg(MI->getOperand(1).getReg()).addMBB(sinkMBB);

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
  BuildMI(*BB, BB->begin(), dl,
          TII.get(XCore::PHI), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(3).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(2).getReg()).addMBB(thisMBB);

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return BB;
}

//===----------------------------------------------------------------------===//
// Target Optimization Hooks
//===----------------------------------------------------------------------===//

SDValue XCoreTargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  DebugLoc dl = N->getDebugLoc();
  switch (N->getOpcode()) {
  default: break;
  case XCoreISD::LADD: {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue N2 = N->getOperand(2);
    ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
    ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
    EVT VT = N0.getValueType();

    // canonicalize constant to RHS
    if (N0C && !N1C)
      return DAG.getNode(XCoreISD::LADD, dl, DAG.getVTList(VT, VT), N1, N0, N2);

    // fold (ladd 0, 0, x) -> 0, x & 1
    if (N0C && N0C->isNullValue() && N1C && N1C->isNullValue()) {
      SDValue Carry = DAG.getConstant(0, VT);
      SDValue Result = DAG.getNode(ISD::AND, dl, VT, N2,
                                   DAG.getConstant(1, VT));
      SDValue Ops [] = { Carry, Result };
      return DAG.getMergeValues(Ops, 2, dl);
    }

    // fold (ladd x, 0, y) -> 0, add x, y iff carry is unused and y has only the
    // low bit set
    if (N1C && N1C->isNullValue() && N->hasNUsesOfValue(0, 0)) {
      APInt KnownZero, KnownOne;
      APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(),
                                         VT.getSizeInBits() - 1);
      DAG.ComputeMaskedBits(N2, KnownZero, KnownOne);
      if ((KnownZero & Mask) == Mask) {
        SDValue Carry = DAG.getConstant(0, VT);
        SDValue Result = DAG.getNode(ISD::ADD, dl, VT, N0, N2);
        SDValue Ops [] = { Carry, Result };
        return DAG.getMergeValues(Ops, 2, dl);
      }
    }
  }
  break;
  case XCoreISD::LSUB: {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue N2 = N->getOperand(2);
    ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
    ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
    EVT VT = N0.getValueType();

    // fold (lsub 0, 0, x) -> x, -x iff x has only the low bit set
    if (N0C && N0C->isNullValue() && N1C && N1C->isNullValue()) {
      APInt KnownZero, KnownOne;
      APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(),
                                         VT.getSizeInBits() - 1);
      DAG.ComputeMaskedBits(N2, KnownZero, KnownOne);
      if ((KnownZero & Mask) == Mask) {
        SDValue Borrow = N2;
        SDValue Result = DAG.getNode(ISD::SUB, dl, VT,
                                     DAG.getConstant(0, VT), N2);
        SDValue Ops [] = { Borrow, Result };
        return DAG.getMergeValues(Ops, 2, dl);
      }
    }

    // fold (lsub x, 0, y) -> 0, sub x, y iff borrow is unused and y has only the
    // low bit set
    if (N1C && N1C->isNullValue() && N->hasNUsesOfValue(0, 0)) {
      APInt KnownZero, KnownOne;
      APInt Mask = APInt::getHighBitsSet(VT.getSizeInBits(),
                                         VT.getSizeInBits() - 1);
      DAG.ComputeMaskedBits(N2, KnownZero, KnownOne);
      if ((KnownZero & Mask) == Mask) {
        SDValue Borrow = DAG.getConstant(0, VT);
        SDValue Result = DAG.getNode(ISD::SUB, dl, VT, N0, N2);
        SDValue Ops [] = { Borrow, Result };
        return DAG.getMergeValues(Ops, 2, dl);
      }
    }
  }
  break;
  case XCoreISD::LMUL: {
    SDValue N0 = N->getOperand(0);
    SDValue N1 = N->getOperand(1);
    SDValue N2 = N->getOperand(2);
    SDValue N3 = N->getOperand(3);
    ConstantSDNode *N0C = dyn_cast<ConstantSDNode>(N0);
    ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1);
    EVT VT = N0.getValueType();
    // Canonicalize multiplicative constant to RHS. If both multiplicative
    // operands are constant canonicalize smallest to RHS.
    if ((N0C && !N1C) ||
        (N0C && N1C && N0C->getZExtValue() < N1C->getZExtValue()))
      return DAG.getNode(XCoreISD::LMUL, dl, DAG.getVTList(VT, VT),
                         N1, N0, N2, N3);

    // lmul(x, 0, a, b)
    if (N1C && N1C->isNullValue()) {
      // If the high result is unused fold to add(a, b)
      if (N->hasNUsesOfValue(0, 0)) {
        SDValue Lo = DAG.getNode(ISD::ADD, dl, VT, N2, N3);
        SDValue Ops [] = { Lo, Lo };
        return DAG.getMergeValues(Ops, 2, dl);
      }
      // Otherwise fold to ladd(a, b, 0)
      return DAG.getNode(XCoreISD::LADD, dl, DAG.getVTList(VT, VT), N2, N3, N1);
    }
  }
  break;
  case ISD::ADD: {
    // Fold 32 bit expressions such as add(add(mul(x,y),a),b) ->
    // lmul(x, y, a, b). The high result of lmul will be ignored.
    // This is only profitable if the intermediate results are unused
    // elsewhere.
    SDValue Mul0, Mul1, Addend0, Addend1;
    if (N->getValueType(0) == MVT::i32 &&
        isADDADDMUL(SDValue(N, 0), Mul0, Mul1, Addend0, Addend1, true)) {
      SDValue Ignored = DAG.getNode(XCoreISD::LMUL, dl,
                                    DAG.getVTList(MVT::i32, MVT::i32), Mul0,
                                    Mul1, Addend0, Addend1);
      SDValue Result(Ignored.getNode(), 1);
      return Result;
    }
    APInt HighMask = APInt::getHighBitsSet(64, 32);
    // Fold 64 bit expression such as add(add(mul(x,y),a),b) ->
    // lmul(x, y, a, b) if all operands are zero-extended. We do this
    // before type legalization as it is messy to match the operands after
    // that.
    if (N->getValueType(0) == MVT::i64 &&
        isADDADDMUL(SDValue(N, 0), Mul0, Mul1, Addend0, Addend1, false) &&
        DAG.MaskedValueIsZero(Mul0, HighMask) &&
        DAG.MaskedValueIsZero(Mul1, HighMask) &&
        DAG.MaskedValueIsZero(Addend0, HighMask) &&
        DAG.MaskedValueIsZero(Addend1, HighMask)) {
      SDValue Mul0L = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                                  Mul0, DAG.getConstant(0, MVT::i32));
      SDValue Mul1L = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                                  Mul1, DAG.getConstant(0, MVT::i32));
      SDValue Addend0L = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                                     Addend0, DAG.getConstant(0, MVT::i32));
      SDValue Addend1L = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, MVT::i32,
                                     Addend1, DAG.getConstant(0, MVT::i32));
      SDValue Hi = DAG.getNode(XCoreISD::LMUL, dl,
                               DAG.getVTList(MVT::i32, MVT::i32), Mul0L, Mul1L,
                               Addend0L, Addend1L);
      SDValue Lo(Hi.getNode(), 1);
      return DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, Lo, Hi);
    }
  }
  break;
  case ISD::STORE: {
    // Replace unaligned store of unaligned load with memmove.
    StoreSDNode *ST  = cast<StoreSDNode>(N);
    if (!DCI.isBeforeLegalize() ||
        allowsUnalignedMemoryAccesses(ST->getMemoryVT()) ||
        ST->isVolatile() || ST->isIndexed()) {
      break;
    }
    SDValue Chain = ST->getChain();

    unsigned StoreBits = ST->getMemoryVT().getStoreSizeInBits();
    if (StoreBits % 8) {
      break;
    }
    unsigned ABIAlignment = getDataLayout()->getABITypeAlignment(
        ST->getMemoryVT().getTypeForEVT(*DCI.DAG.getContext()));
    unsigned Alignment = ST->getAlignment();
    if (Alignment >= ABIAlignment) {
      break;
    }

    if (LoadSDNode *LD = dyn_cast<LoadSDNode>(ST->getValue())) {
      if (LD->hasNUsesOfValue(1, 0) && ST->getMemoryVT() == LD->getMemoryVT() &&
        LD->getAlignment() == Alignment &&
        !LD->isVolatile() && !LD->isIndexed() &&
        Chain.reachesChainWithoutSideEffects(SDValue(LD, 1))) {
        return DAG.getMemmove(Chain, dl, ST->getBasePtr(),
                              LD->getBasePtr(),
                              DAG.getConstant(StoreBits/8, MVT::i32),
                              Alignment, false, ST->getPointerInfo(),
                              LD->getPointerInfo());
      }
    }
    break;
  }
  }
  return SDValue();
}

void XCoreTargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                         APInt &KnownZero,
                                                         APInt &KnownOne,
                                                         const SelectionDAG &DAG,
                                                         unsigned Depth) const {
  KnownZero = KnownOne = APInt(KnownZero.getBitWidth(), 0);
  switch (Op.getOpcode()) {
  default: break;
  case XCoreISD::LADD:
  case XCoreISD::LSUB:
    if (Op.getResNo() == 0) {
      // Top bits of carry / borrow are clear.
      KnownZero = APInt::getHighBitsSet(KnownZero.getBitWidth(),
                                        KnownZero.getBitWidth() - 1);
    }
    break;
  }
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
                                              Type *Ty) const {
  if (Ty->getTypeID() == Type::VoidTyID)
    return AM.Scale == 0 && isImmUs(AM.BaseOffs) && isImmUs4(AM.BaseOffs);

  const DataLayout *TD = TM.getDataLayout();
  unsigned Size = TD->getTypeAllocSize(Ty);
  if (AM.BaseGV) {
    return Size >= 4 && !AM.HasBaseReg && AM.Scale == 0 &&
                 AM.BaseOffs%4 == 0;
  }

  switch (Size) {
  case 1:
    // reg + imm
    if (AM.Scale == 0) {
      return isImmUs(AM.BaseOffs);
    }
    // reg + reg
    return AM.Scale == 1 && AM.BaseOffs == 0;
  case 2:
  case 3:
    // reg + imm
    if (AM.Scale == 0) {
      return isImmUs2(AM.BaseOffs);
    }
    // reg + reg<<1
    return AM.Scale == 2 && AM.BaseOffs == 0;
  default:
    // reg + imm
    if (AM.Scale == 0) {
      return isImmUs4(AM.BaseOffs);
    }
    // reg + reg<<2
    return AM.Scale == 4 && AM.BaseOffs == 0;
  }
}

//===----------------------------------------------------------------------===//
//                           XCore Inline Assembly Support
//===----------------------------------------------------------------------===//

std::pair<unsigned, const TargetRegisterClass*>
XCoreTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint,
                             EVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default : break;
    case 'r':
      return std::make_pair(0U, &XCore::GRRegsRegClass);
    }
  }
  // Use the default implementation in TargetLowering to convert the register
  // constraint into a member of a register class.
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}
