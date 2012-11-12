//===-- HexagonISelLowering.cpp - Hexagon DAG Lowering Implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interfaces that Hexagon uses to lower LLVM code
// into a selection DAG.
//
//===----------------------------------------------------------------------===//

#include "HexagonISelLowering.h"
#include "HexagonTargetMachine.h"
#include "HexagonMachineFunctionInfo.h"
#include "HexagonTargetObjectFile.h"
#include "HexagonSubtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/InlineAsm.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

const unsigned Hexagon_MAX_RET_SIZE = 64;

static cl::opt<bool>
EmitJumpTables("hexagon-emit-jump-tables", cl::init(true), cl::Hidden,
               cl::desc("Control jump table emission on Hexagon target"));

int NumNamedVarArgParams = -1;

// Implement calling convention for Hexagon.
static bool
CC_Hexagon(unsigned ValNo, MVT ValVT,
           MVT LocVT, CCValAssign::LocInfo LocInfo,
           ISD::ArgFlagsTy ArgFlags, CCState &State);

static bool
CC_Hexagon32(unsigned ValNo, MVT ValVT,
             MVT LocVT, CCValAssign::LocInfo LocInfo,
             ISD::ArgFlagsTy ArgFlags, CCState &State);

static bool
CC_Hexagon64(unsigned ValNo, MVT ValVT,
             MVT LocVT, CCValAssign::LocInfo LocInfo,
             ISD::ArgFlagsTy ArgFlags, CCState &State);

static bool
RetCC_Hexagon(unsigned ValNo, MVT ValVT,
              MVT LocVT, CCValAssign::LocInfo LocInfo,
              ISD::ArgFlagsTy ArgFlags, CCState &State);

static bool
RetCC_Hexagon32(unsigned ValNo, MVT ValVT,
                MVT LocVT, CCValAssign::LocInfo LocInfo,
                ISD::ArgFlagsTy ArgFlags, CCState &State);

static bool
RetCC_Hexagon64(unsigned ValNo, MVT ValVT,
                MVT LocVT, CCValAssign::LocInfo LocInfo,
                ISD::ArgFlagsTy ArgFlags, CCState &State);

static bool
CC_Hexagon_VarArg (unsigned ValNo, MVT ValVT,
            MVT LocVT, CCValAssign::LocInfo LocInfo,
            ISD::ArgFlagsTy ArgFlags, CCState &State) {

  // NumNamedVarArgParams can not be zero for a VarArg function.
  assert ( (NumNamedVarArgParams > 0) &&
           "NumNamedVarArgParams is not bigger than zero.");

  if ( (int)ValNo < NumNamedVarArgParams ) {
    // Deal with named arguments.
    return CC_Hexagon(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State);
  }

  // Deal with un-named arguments.
  unsigned ofst;
  if (ArgFlags.isByVal()) {
    // If pass-by-value, the size allocated on stack is decided
    // by ArgFlags.getByValSize(), not by the size of LocVT.
    assert ((ArgFlags.getByValSize() > 8) &&
            "ByValSize must be bigger than 8 bytes");
    ofst = State.AllocateStack(ArgFlags.getByValSize(), 4);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, ofst, LocVT, LocInfo));
    return false;
  }
  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    ofst = State.AllocateStack(4, 4);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, ofst, LocVT, LocInfo));
    return false;
  }
  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    ofst = State.AllocateStack(8, 8);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, ofst, LocVT, LocInfo));
    return false;
  }
  llvm_unreachable(0);
}


static bool
CC_Hexagon (unsigned ValNo, MVT ValVT,
            MVT LocVT, CCValAssign::LocInfo LocInfo,
            ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (ArgFlags.isByVal()) {
    // Passed on stack.
    assert ((ArgFlags.getByValSize() > 8) &&
            "ByValSize must be bigger than 8 bytes");
    unsigned Offset = State.AllocateStack(ArgFlags.getByValSize(), 4);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
    return false;
  }

  if (LocVT == MVT::i1 || LocVT == MVT::i8 || LocVT == MVT::i16) {
    LocVT = MVT::i32;
    ValVT = MVT::i32;
    if (ArgFlags.isSExt())
      LocInfo = CCValAssign::SExt;
    else if (ArgFlags.isZExt())
      LocInfo = CCValAssign::ZExt;
    else
      LocInfo = CCValAssign::AExt;
  }

  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    if (!CC_Hexagon32(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State))
      return false;
  }

  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    if (!CC_Hexagon64(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State))
      return false;
  }

  return true;  // CC didn't match.
}


static bool CC_Hexagon32(unsigned ValNo, MVT ValVT,
                         MVT LocVT, CCValAssign::LocInfo LocInfo,
                         ISD::ArgFlagsTy ArgFlags, CCState &State) {

  static const uint16_t RegList[] = {
    Hexagon::R0, Hexagon::R1, Hexagon::R2, Hexagon::R3, Hexagon::R4,
    Hexagon::R5
  };
  if (unsigned Reg = State.AllocateReg(RegList, 6)) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  unsigned Offset = State.AllocateStack(4, 4);
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  return false;
}

static bool CC_Hexagon64(unsigned ValNo, MVT ValVT,
                         MVT LocVT, CCValAssign::LocInfo LocInfo,
                         ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (unsigned Reg = State.AllocateReg(Hexagon::D0)) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  static const uint16_t RegList1[] = {
    Hexagon::D1, Hexagon::D2
  };
  static const uint16_t RegList2[] = {
    Hexagon::R1, Hexagon::R3
  };
  if (unsigned Reg = State.AllocateReg(RegList1, RegList2, 2)) {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
    return false;
  }

  unsigned Offset = State.AllocateStack(8, 8, Hexagon::D2);
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  return false;
}

static bool RetCC_Hexagon(unsigned ValNo, MVT ValVT,
                          MVT LocVT, CCValAssign::LocInfo LocInfo,
                          ISD::ArgFlagsTy ArgFlags, CCState &State) {


  if (LocVT == MVT::i1 ||
      LocVT == MVT::i8 ||
      LocVT == MVT::i16) {
    LocVT = MVT::i32;
    ValVT = MVT::i32;
    if (ArgFlags.isSExt())
      LocInfo = CCValAssign::SExt;
    else if (ArgFlags.isZExt())
      LocInfo = CCValAssign::ZExt;
    else
      LocInfo = CCValAssign::AExt;
  }

  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    if (!RetCC_Hexagon32(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State))
    return false;
  }

  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    if (!RetCC_Hexagon64(ValNo, ValVT, LocVT, LocInfo, ArgFlags, State))
    return false;
  }

  return true;  // CC didn't match.
}

static bool RetCC_Hexagon32(unsigned ValNo, MVT ValVT,
                            MVT LocVT, CCValAssign::LocInfo LocInfo,
                            ISD::ArgFlagsTy ArgFlags, CCState &State) {

  if (LocVT == MVT::i32 || LocVT == MVT::f32) {
    if (unsigned Reg = State.AllocateReg(Hexagon::R0)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  unsigned Offset = State.AllocateStack(4, 4);
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  return false;
}

static bool RetCC_Hexagon64(unsigned ValNo, MVT ValVT,
                            MVT LocVT, CCValAssign::LocInfo LocInfo,
                            ISD::ArgFlagsTy ArgFlags, CCState &State) {
  if (LocVT == MVT::i64 || LocVT == MVT::f64) {
    if (unsigned Reg = State.AllocateReg(Hexagon::D0)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return false;
    }
  }

  unsigned Offset = State.AllocateStack(8, 8);
  State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  return false;
}

SDValue
HexagonTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG)
const {
  return SDValue();
}

/// CreateCopyOfByValArgument - Make a copy of an aggregate at address specified
/// by "Src" to address "Dst" of size "Size".  Alignment information is
/// specified by the specific parameter attribute. The copy will be passed as
/// a byval function parameter.  Sometimes what we are copying is the end of a
/// larger object, the part that does not fit in registers.
static SDValue
CreateCopyOfByValArgument(SDValue Src, SDValue Dst, SDValue Chain,
                          ISD::ArgFlagsTy Flags, SelectionDAG &DAG,
                          DebugLoc dl) {

  SDValue SizeNode = DAG.getConstant(Flags.getByValSize(), MVT::i32);
  return DAG.getMemcpy(Chain, dl, Dst, Src, SizeNode, Flags.getByValAlign(),
                       /*isVolatile=*/false, /*AlwaysInline=*/false,
                       MachinePointerInfo(), MachinePointerInfo());
}


// LowerReturn - Lower ISD::RET. If a struct is larger than 8 bytes and is
// passed by value, the function prototype is modified to return void and
// the value is stored in memory pointed by a pointer passed by caller.
SDValue
HexagonTargetLowering::LowerReturn(SDValue Chain,
                                   CallingConv::ID CallConv, bool isVarArg,
                                   const SmallVectorImpl<ISD::OutputArg> &Outs,
                                   const SmallVectorImpl<SDValue> &OutVals,
                                   DebugLoc dl, SelectionDAG &DAG) const {

  // CCValAssign - represent the assignment of the return value to locations.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  // Analyze return values of ISD::RET
  CCInfo.AnalyzeReturn(Outs, RetCC_Hexagon);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Flag;
  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), OutVals[i], Flag);

    // Guarantee that all emitted copies are stuck together with flags.
    Flag = Chain.getValue(1);
  }

  if (Flag.getNode())
    return DAG.getNode(HexagonISD::RET_FLAG, dl, MVT::Other, Chain, Flag);

  return DAG.getNode(HexagonISD::RET_FLAG, dl, MVT::Other, Chain);
}




/// LowerCallResult - Lower the result values of an ISD::CALL into the
/// appropriate copies out of appropriate physical registers.  This assumes that
/// Chain/InFlag are the input chain/flag to use, and that TheCall is the call
/// being lowered. Returns a SDNode with the same number of values as the
/// ISD::CALL.
SDValue
HexagonTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                       CallingConv::ID CallConv, bool isVarArg,
                                       const
                                       SmallVectorImpl<ISD::InputArg> &Ins,
                                       DebugLoc dl, SelectionDAG &DAG,
                                       SmallVectorImpl<SDValue> &InVals,
                                       const SmallVectorImpl<SDValue> &OutVals,
                                       SDValue Callee) const {

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;

  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  CCInfo.AnalyzeCallResult(Ins, RetCC_Hexagon);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    Chain = DAG.getCopyFromReg(Chain, dl,
                               RVLocs[i].getLocReg(),
                               RVLocs[i].getValVT(), InFlag).getValue(1);
    InFlag = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

/// LowerCall - Functions arguments are copied from virtual regs to
/// (physical regs)/(stack frame), CALLSEQ_START and CALLSEQ_END are emitted.
SDValue
HexagonTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
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

  bool IsStructRet    = (Outs.empty()) ? false : Outs[0].Flags.isSRet();

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  // Check for varargs.
  NumNamedVarArgParams = -1;
  if (GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Callee))
  {
    const Function* CalleeFn = NULL;
    Callee = DAG.getTargetGlobalAddress(GA->getGlobal(), dl, MVT::i32);
    if ((CalleeFn = dyn_cast<Function>(GA->getGlobal())))
    {
      // If a function has zero args and is a vararg function, that's
      // disallowed so it must be an undeclared function.  Do not assume
      // varargs if the callee is undefined.
      if (CalleeFn->isVarArg() &&
          CalleeFn->getFunctionType()->getNumParams() != 0) {
        NumNamedVarArgParams = CalleeFn->getFunctionType()->getNumParams();
      }
    }
  }

  if (NumNamedVarArgParams > 0)
    CCInfo.AnalyzeCallOperands(Outs, CC_Hexagon_VarArg);
  else
    CCInfo.AnalyzeCallOperands(Outs, CC_Hexagon);


  if(isTailCall) {
    bool StructAttrFlag =
      DAG.getMachineFunction().getFunction()->hasStructRetAttr();
    isTailCall = IsEligibleForTailCallOptimization(Callee, CallConv,
                                                   isVarArg, IsStructRet,
                                                   StructAttrFlag,
                                                   Outs, OutVals, Ins, DAG);
    for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i){
      CCValAssign &VA = ArgLocs[i];
      if (VA.isMemLoc()) {
        isTailCall = false;
        break;
      }
    }
    if (isTailCall) {
      DEBUG(dbgs () << "Eligible for Tail Call\n");
    } else {
      DEBUG(dbgs () <<
            "Argument must be passed on stack. Not eligible for Tail Call\n");
    }
  }
  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();
  SmallVector<std::pair<unsigned, SDValue>, 16> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  SDValue StackPtr =
    DAG.getCopyFromReg(Chain, dl, TM.getRegisterInfo()->getStackRegister(),
                       getPointerTy());

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
      default:
        // Loc info must be one of Full, SExt, ZExt, or AExt.
        llvm_unreachable("Unknown loc info!");
      case CCValAssign::Full:
        break;
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

    if (VA.isMemLoc()) {
      unsigned LocMemOffset = VA.getLocMemOffset();
      SDValue PtrOff = DAG.getConstant(LocMemOffset, StackPtr.getValueType());
      PtrOff = DAG.getNode(ISD::ADD, dl, MVT::i32, StackPtr, PtrOff);

      if (Flags.isByVal()) {
        // The argument is a struct passed by value. According to LLVM, "Arg"
        // is is pointer.
        MemOpChains.push_back(CreateCopyOfByValArgument(Arg, PtrOff, Chain,
                                                        Flags, DAG, dl));
      } else {
        // The argument is not passed by value. "Arg" is a buildin type. It is
        // not a pointer.
        MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                           MachinePointerInfo(),false, false,
                                           0));
      }
      continue;
    }

    // Arguments that can be passed on register must be kept at RegsToPass
    // vector.
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    }
  }

  // Transform all store nodes into one single node because all store
  // nodes are independent of each other.
  if (!MemOpChains.empty()) {
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &MemOpChains[0],
                        MemOpChains.size());
  }

  if (!isTailCall)
    Chain = DAG.getCALLSEQ_START(Chain, DAG.getConstant(NumBytes,
                                                        getPointerTy(), true));

  // Build a sequence of copy-to-reg nodes chained together with token
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emitted instructions must be
  // stuck together.
  SDValue InFlag;
  if (!isTailCall) {
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
      Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                               RegsToPass[i].second, InFlag);
      InFlag = Chain.getValue(1);
    }
  }

  // For tail calls lower the arguments to the 'real' stack slot.
  if (isTailCall) {
    // Force all the incoming stack arguments to be loaded from the stack
    // before any new outgoing arguments are stored to the stack, because the
    // outgoing stack slots may alias the incoming argument stack slots, and
    // the alias isn't otherwise explicit. This is slightly more conservative
    // than necessary, because it means that each store effectively depends
    // on every argument instead of just those arguments it would clobber.
    //
    // Do not flag preceding copytoreg stuff together with the following stuff.
    InFlag = SDValue();
    for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
      Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                               RegsToPass[i].second, InFlag);
      InFlag = Chain.getValue(1);
    }
    InFlag =SDValue();
  }

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  if (flag_aligned_memcpy) {
    const char *MemcpyName =
      "__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes";
    Callee =
      DAG.getTargetExternalSymbol(MemcpyName, getPointerTy());
    flag_aligned_memcpy = false;
  } else if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl, getPointerTy());
  } else if (ExternalSymbolSDNode *S =
             dyn_cast<ExternalSymbolSDNode>(Callee)) {
    Callee = DAG.getTargetExternalSymbol(S->getSymbol(), getPointerTy());
  }

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

  if (InFlag.getNode()) {
    Ops.push_back(InFlag);
  }

  if (isTailCall)
    return DAG.getNode(HexagonISD::TC_RETURN, dl, NodeTys, &Ops[0], Ops.size());

  Chain = DAG.getNode(HexagonISD::CALL, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg, Ins, dl, DAG,
                         InVals, OutVals, Callee);
}

static bool getIndexedAddressParts(SDNode *Ptr, EVT VT,
                                   bool isSEXTLoad, SDValue &Base,
                                   SDValue &Offset, bool &isInc,
                                   SelectionDAG &DAG) {
  if (Ptr->getOpcode() != ISD::ADD)
  return false;

  if (VT == MVT::i64 || VT == MVT::i32 || VT == MVT::i16 || VT == MVT::i8) {
    isInc = (Ptr->getOpcode() == ISD::ADD);
    Base = Ptr->getOperand(0);
    Offset = Ptr->getOperand(1);
    // Ensure that Offset is a constant.
    return (isa<ConstantSDNode>(Offset));
  }

  return false;
}

// TODO: Put this function along with the other isS* functions in
// HexagonISelDAGToDAG.cpp into a common file. Or better still, use the
// functions defined in HexagonImmediates.td.
static bool Is_PostInc_S4_Offset(SDNode * S, int ShiftAmount) {
  ConstantSDNode *N = cast<ConstantSDNode>(S);

  // immS4 predicate - True if the immediate fits in a 4-bit sign extended.
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  int64_t m = 0;
  if (ShiftAmount > 0) {
    m = v % ShiftAmount;
    v = v >> ShiftAmount;
  }
  return (v <= 7) && (v >= -8) && (m == 0);
}

/// getPostIndexedAddressParts - returns true by value, base pointer and
/// offset pointer and addressing mode by reference if this node can be
/// combined with a load / store to form a post-indexed load / store.
bool HexagonTargetLowering::getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                                       SDValue &Base,
                                                       SDValue &Offset,
                                                       ISD::MemIndexedMode &AM,
                                                       SelectionDAG &DAG) const
{
  EVT VT;
  SDValue Ptr;
  bool isSEXTLoad = false;

  if (LoadSDNode *LD = dyn_cast<LoadSDNode>(N)) {
    VT  = LD->getMemoryVT();
    isSEXTLoad = LD->getExtensionType() == ISD::SEXTLOAD;
  } else if (StoreSDNode *ST = dyn_cast<StoreSDNode>(N)) {
    VT  = ST->getMemoryVT();
    if (ST->getValue().getValueType() == MVT::i64 && ST->isTruncatingStore()) {
      return false;
    }
  } else {
    return false;
  }

  bool isInc = false;
  bool isLegal = getIndexedAddressParts(Op, VT, isSEXTLoad, Base, Offset,
                                        isInc, DAG);
  // ShiftAmount = number of left-shifted bits in the Hexagon instruction.
  int ShiftAmount = VT.getSizeInBits() / 16;
  if (isLegal && Is_PostInc_S4_Offset(Offset.getNode(), ShiftAmount)) {
    AM = isInc ? ISD::POST_INC : ISD::POST_DEC;
    return true;
  }

  return false;
}

SDValue HexagonTargetLowering::LowerINLINEASM(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  MachineFunction &MF = DAG.getMachineFunction();
  HexagonMachineFunctionInfo *FuncInfo =
    MF.getInfo<HexagonMachineFunctionInfo>();
  switch (Node->getOpcode()) {
    case ISD::INLINEASM: {
      unsigned NumOps = Node->getNumOperands();
      if (Node->getOperand(NumOps-1).getValueType() == MVT::Glue)
        --NumOps;  // Ignore the flag operand.

      for (unsigned i = InlineAsm::Op_FirstOperand; i != NumOps;) {
        if (FuncInfo->hasClobberLR())
          break;
        unsigned Flags =
          cast<ConstantSDNode>(Node->getOperand(i))->getZExtValue();
        unsigned NumVals = InlineAsm::getNumOperandRegisters(Flags);
        ++i;  // Skip the ID value.

        switch (InlineAsm::getKind(Flags)) {
        default: llvm_unreachable("Bad flags!");
          case InlineAsm::Kind_RegDef:
          case InlineAsm::Kind_RegUse:
          case InlineAsm::Kind_Imm:
          case InlineAsm::Kind_Clobber:
          case InlineAsm::Kind_Mem: {
            for (; NumVals; --NumVals, ++i) {}
            break;
          }
          case InlineAsm::Kind_RegDefEarlyClobber: {
            for (; NumVals; --NumVals, ++i) {
              unsigned Reg =
                cast<RegisterSDNode>(Node->getOperand(i))->getReg();

              // Check it to be lr
              if (Reg == TM.getRegisterInfo()->getRARegister()) {
                FuncInfo->setHasClobberLR(true);
                break;
              }
            }
            break;
          }
        }
      }
    }
  } // Node->getOpcode
  return Op;
}


//
// Taken from the XCore backend.
//
SDValue HexagonTargetLowering::
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

  // Mark all jump table targets as address taken.
  const std::vector<MachineJumpTableEntry> &JTE = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JTE[JTI].MBBs;
  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    MBB->setHasAddressTaken();
    // This line is needed to set the hasAddressTaken flag on the BasicBlock
    // object.
    BlockAddress::get(const_cast<BasicBlock *>(MBB->getBasicBlock()));
  }

  SDValue JumpTableBase = DAG.getNode(HexagonISD::WrapperJT, dl,
                                      getPointerTy(), TargetJT);
  SDValue ShiftIndex = DAG.getNode(ISD::SHL, dl, MVT::i32, Index,
                                   DAG.getConstant(2, MVT::i32));
  SDValue JTAddress = DAG.getNode(ISD::ADD, dl, MVT::i32, JumpTableBase,
                                  ShiftIndex);
  SDValue LoadTarget = DAG.getLoad(MVT::i32, dl, Chain, JTAddress,
                                   MachinePointerInfo(), false, false, false,
                                   0);
  return DAG.getNode(HexagonISD::BR_JT, dl, MVT::Other, Chain, LoadTarget);
}


SDValue
HexagonTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Size = Op.getOperand(1);
  DebugLoc dl = Op.getDebugLoc();

  unsigned SPReg = getStackPointerRegisterToSaveRestore();

  // Get a reference to the stack pointer.
  SDValue StackPointer = DAG.getCopyFromReg(Chain, dl, SPReg, MVT::i32);

  // Subtract the dynamic size from the actual stack size to
  // obtain the new stack size.
  SDValue Sub = DAG.getNode(ISD::SUB, dl, MVT::i32, StackPointer, Size);

  //
  // For Hexagon, the outgoing memory arguments area should be on top of the
  // alloca area on the stack i.e., the outgoing memory arguments should be
  // at a lower address than the alloca area. Move the alloca area down the
  // stack by adding back the space reserved for outgoing arguments to SP
  // here.
  //
  // We do not know what the size of the outgoing args is at this point.
  // So, we add a pseudo instruction ADJDYNALLOC that will adjust the
  // stack pointer. We patch this instruction with the correct, known
  // offset in emitPrologue().
  //
  // Use a placeholder immediate (zero) for now. This will be patched up
  // by emitPrologue().
  SDValue ArgAdjust = DAG.getNode(HexagonISD::ADJDYNALLOC, dl,
                                  MVT::i32,
                                  Sub,
                                  DAG.getConstant(0, MVT::i32));

  // The Sub result contains the new stack start address, so it
  // must be placed in the stack pointer register.
  SDValue CopyChain = DAG.getCopyToReg(Chain, dl,
                                       TM.getRegisterInfo()->getStackRegister(),
                                       Sub);

  SDValue Ops[2] = { ArgAdjust, CopyChain };
  return DAG.getMergeValues(Ops, 2, dl);
}

SDValue
HexagonTargetLowering::LowerFormalArguments(SDValue Chain,
                                            CallingConv::ID CallConv,
                                            bool isVarArg,
                                            const
                                            SmallVectorImpl<ISD::InputArg> &Ins,
                                            DebugLoc dl, SelectionDAG &DAG,
                                            SmallVectorImpl<SDValue> &InVals)
const {

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  HexagonMachineFunctionInfo *FuncInfo =
    MF.getInfo<HexagonMachineFunctionInfo>();


  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  CCInfo.AnalyzeFormalArguments(Ins, CC_Hexagon);

  // For LLVM, in the case when returning a struct by value (>8byte),
  // the first argument is a pointer that points to the location on caller's
  // stack where the return value will be stored. For Hexagon, the location on
  // caller's stack is passed only when the struct size is smaller than (and
  // equal to) 8 bytes. If not, no address will be passed into callee and
  // callee return the result direclty through R0/R1.

  SmallVector<SDValue, 4> MemOps;

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    ISD::ArgFlagsTy Flags = Ins[i].Flags;
    unsigned ObjSize;
    unsigned StackLocation;
    int FI;

    if (   (VA.isRegLoc() && !Flags.isByVal())
        || (VA.isRegLoc() && Flags.isByVal() && Flags.getByValSize() > 8)) {
      // Arguments passed in registers
      // 1. int, long long, ptr args that get allocated in register.
      // 2. Large struct that gets an register to put its address in.
      EVT RegVT = VA.getLocVT();
      if (RegVT == MVT::i8 || RegVT == MVT::i16 ||
          RegVT == MVT::i32 || RegVT == MVT::f32) {
        unsigned VReg =
          RegInfo.createVirtualRegister(&Hexagon::IntRegsRegClass);
        RegInfo.addLiveIn(VA.getLocReg(), VReg);
        InVals.push_back(DAG.getCopyFromReg(Chain, dl, VReg, RegVT));
      } else if (RegVT == MVT::i64) {
        unsigned VReg =
          RegInfo.createVirtualRegister(&Hexagon::DoubleRegsRegClass);
        RegInfo.addLiveIn(VA.getLocReg(), VReg);
        InVals.push_back(DAG.getCopyFromReg(Chain, dl, VReg, RegVT));
      } else {
        assert (0);
      }
    } else if (VA.isRegLoc() && Flags.isByVal() && Flags.getByValSize() <= 8) {
      assert (0 && "ByValSize must be bigger than 8 bytes");
    } else {
      // Sanity check.
      assert(VA.isMemLoc());

      if (Flags.isByVal()) {
        // If it's a byval parameter, then we need to compute the
        // "real" size, not the size of the pointer.
        ObjSize = Flags.getByValSize();
      } else {
        ObjSize = VA.getLocVT().getStoreSizeInBits() >> 3;
      }

      StackLocation = HEXAGON_LRFP_SIZE + VA.getLocMemOffset();
      // Create the frame index object for this incoming parameter...
      FI = MFI->CreateFixedObject(ObjSize, StackLocation, true);

      // Create the SelectionDAG nodes cordl, responding to a load
      // from this parameter.
      SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);

      if (Flags.isByVal()) {
        // If it's a pass-by-value aggregate, then do not dereference the stack
        // location. Instead, we should generate a reference to the stack
        // location.
        InVals.push_back(FIN);
      } else {
        InVals.push_back(DAG.getLoad(VA.getLocVT(), dl, Chain, FIN,
                                     MachinePointerInfo(), false, false,
                                     false, 0));
      }
    }
  }

  if (!MemOps.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &MemOps[0],
                        MemOps.size());

  if (isVarArg) {
    // This will point to the next argument passed via stack.
    int FrameIndex = MFI->CreateFixedObject(Hexagon_PointerSize,
                                            HEXAGON_LRFP_SIZE +
                                            CCInfo.getNextStackOffset(),
                                            true);
    FuncInfo->setVarArgsFrameIndex(FrameIndex);
  }

  return Chain;
}

SDValue
HexagonTargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  // VASTART stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  MachineFunction &MF = DAG.getMachineFunction();
  HexagonMachineFunctionInfo *QFI = MF.getInfo<HexagonMachineFunctionInfo>();
  SDValue Addr = DAG.getFrameIndex(QFI->getVarArgsFrameIndex(), MVT::i32);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), Op.getDebugLoc(), Addr,
                      Op.getOperand(1), MachinePointerInfo(SV), false,
                      false, 0);
}

SDValue
HexagonTargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue CC = Op.getOperand(4);
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);
  DebugLoc dl = Op.getDebugLoc();
  SDNode* OpNode = Op.getNode();
  EVT SVT = OpNode->getValueType(0);

  SDValue Cond = DAG.getNode(ISD::SETCC, dl, MVT::i1, LHS, RHS, CC);
  return DAG.getNode(ISD::SELECT, dl, SVT, Cond, TrueVal, FalseVal);
}

SDValue
HexagonTargetLowering::LowerConstantPool(SDValue Op, SelectionDAG &DAG) const {
  EVT ValTy = Op.getValueType();

  DebugLoc dl = Op.getDebugLoc();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  SDValue Res;
  if (CP->isMachineConstantPoolEntry())
    Res = DAG.getTargetConstantPool(CP->getMachineCPVal(), ValTy,
                                    CP->getAlignment());
  else
    Res = DAG.getTargetConstantPool(CP->getConstVal(), ValTy,
                                    CP->getAlignment());
  return DAG.getNode(HexagonISD::CONST32, dl, ValTy, Res);
}

SDValue
HexagonTargetLowering::LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const {
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MFI->setReturnAddressIsTaken(true);

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  if (Depth) {
    SDValue FrameAddr = LowerFRAMEADDR(Op, DAG);
    SDValue Offset = DAG.getConstant(4, MVT::i32);
    return DAG.getLoad(VT, dl, DAG.getEntryNode(),
                       DAG.getNode(ISD::ADD, dl, VT, FrameAddr, Offset),
                       MachinePointerInfo(), false, false, false, 0);
  }

  // Return LR, which contains the return address. Mark it an implicit live-in.
  unsigned Reg = MF.addLiveIn(TRI->getRARegister(), getRegClassFor(MVT::i32));
  return DAG.getCopyFromReg(DAG.getEntryNode(), dl, Reg, VT);
}

SDValue
HexagonTargetLowering::LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const {
  const HexagonRegisterInfo  *TRI = TM.getRegisterInfo();
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setFrameAddressIsTaken(true);

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  SDValue FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl,
                                         TRI->getFrameRegister(), VT);
  while (Depth--)
    FrameAddr = DAG.getLoad(VT, dl, DAG.getEntryNode(), FrameAddr,
                            MachinePointerInfo(),
                            false, false, false, 0);
  return FrameAddr;
}


SDValue HexagonTargetLowering::LowerMEMBARRIER(SDValue Op,
                                               SelectionDAG& DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  return DAG.getNode(HexagonISD::BARRIER, dl, MVT::Other,  Op.getOperand(0));
}


SDValue HexagonTargetLowering::LowerATOMIC_FENCE(SDValue Op,
                                                 SelectionDAG& DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  return DAG.getNode(HexagonISD::BARRIER, dl, MVT::Other, Op.getOperand(0));
}


SDValue HexagonTargetLowering::LowerGLOBALADDRESS(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDValue Result;
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  int64_t Offset = cast<GlobalAddressSDNode>(Op)->getOffset();
  DebugLoc dl = Op.getDebugLoc();
  Result = DAG.getTargetGlobalAddress(GV, dl, getPointerTy(), Offset);

  HexagonTargetObjectFile &TLOF =
    (HexagonTargetObjectFile&)getObjFileLowering();
  if (TLOF.IsGlobalInSmallSection(GV, getTargetMachine())) {
    return DAG.getNode(HexagonISD::CONST32_GP, dl, getPointerTy(), Result);
  }

  return DAG.getNode(HexagonISD::CONST32, dl, getPointerTy(), Result);
}

//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

HexagonTargetLowering::HexagonTargetLowering(HexagonTargetMachine
                                             &targetmachine)
  : TargetLowering(targetmachine, new HexagonTargetObjectFile()),
    TM(targetmachine) {

    const HexagonRegisterInfo* QRI = TM.getRegisterInfo();

    // Set up the register classes.
    addRegisterClass(MVT::i32, &Hexagon::IntRegsRegClass);
    addRegisterClass(MVT::i64, &Hexagon::DoubleRegsRegClass);

    if (QRI->Subtarget.hasV5TOps()) {
      addRegisterClass(MVT::f32, &Hexagon::IntRegsRegClass);
      addRegisterClass(MVT::f64, &Hexagon::DoubleRegsRegClass);
    }

    addRegisterClass(MVT::i1, &Hexagon::PredRegsRegClass);

    computeRegisterProperties();

    // Align loop entry
    setPrefLoopAlignment(4);

    // Limits for inline expansion of memcpy/memmove
    maxStoresPerMemcpy = 6;
    maxStoresPerMemmove = 6;

    //
    // Library calls for unsupported operations
    //

    setLibcallName(RTLIB::SINTTOFP_I128_F64, "__hexagon_floattidf");
    setLibcallName(RTLIB::SINTTOFP_I128_F32, "__hexagon_floattisf");

    setLibcallName(RTLIB::FPTOUINT_F32_I128, "__hexagon_fixunssfti");
    setLibcallName(RTLIB::FPTOUINT_F64_I128, "__hexagon_fixunsdfti");

    setLibcallName(RTLIB::FPTOSINT_F32_I128, "__hexagon_fixsfti");
    setLibcallName(RTLIB::FPTOSINT_F64_I128, "__hexagon_fixdfti");

    setLibcallName(RTLIB::SDIV_I32, "__hexagon_divsi3");
    setOperationAction(ISD::SDIV,  MVT::i32, Expand);
    setLibcallName(RTLIB::SREM_I32, "__hexagon_umodsi3");
    setOperationAction(ISD::SREM,  MVT::i32, Expand);

    setLibcallName(RTLIB::SDIV_I64, "__hexagon_divdi3");
    setOperationAction(ISD::SDIV,  MVT::i64, Expand);
    setLibcallName(RTLIB::SREM_I64, "__hexagon_moddi3");
    setOperationAction(ISD::SREM,  MVT::i64, Expand);

    setLibcallName(RTLIB::UDIV_I32, "__hexagon_udivsi3");
    setOperationAction(ISD::UDIV,  MVT::i32, Expand);

    setLibcallName(RTLIB::UDIV_I64, "__hexagon_udivdi3");
    setOperationAction(ISD::UDIV,  MVT::i64, Expand);

    setLibcallName(RTLIB::UREM_I32, "__hexagon_umodsi3");
    setOperationAction(ISD::UREM,  MVT::i32, Expand);

    setLibcallName(RTLIB::UREM_I64, "__hexagon_umoddi3");
    setOperationAction(ISD::UREM,  MVT::i64, Expand);

    setLibcallName(RTLIB::DIV_F32, "__hexagon_divsf3");
    setOperationAction(ISD::FDIV,  MVT::f32, Expand);

    setLibcallName(RTLIB::DIV_F64, "__hexagon_divdf3");
    setOperationAction(ISD::FDIV,  MVT::f64, Expand);

    setOperationAction(ISD::FSQRT,  MVT::f32, Expand);
    setOperationAction(ISD::FSQRT,  MVT::f64, Expand);
    setOperationAction(ISD::FSIN,  MVT::f32, Expand);
    setOperationAction(ISD::FSIN,  MVT::f64, Expand);

    if (QRI->Subtarget.hasV5TOps()) {
      // Hexagon V5 Support.
      setOperationAction(ISD::FADD,       MVT::f32, Legal);
      setOperationAction(ISD::FADD,       MVT::f64, Legal);
      setOperationAction(ISD::FP_EXTEND,  MVT::f32, Legal);
      setCondCodeAction(ISD::SETOEQ,      MVT::f32, Legal);
      setCondCodeAction(ISD::SETOEQ,      MVT::f64, Legal);
      setCondCodeAction(ISD::SETUEQ,      MVT::f32, Legal);
      setCondCodeAction(ISD::SETUEQ,      MVT::f64, Legal);

      setCondCodeAction(ISD::SETOGE,      MVT::f32, Legal);
      setCondCodeAction(ISD::SETOGE,      MVT::f64, Legal);
      setCondCodeAction(ISD::SETUGE,      MVT::f32, Legal);
      setCondCodeAction(ISD::SETUGE,      MVT::f64, Legal);

      setCondCodeAction(ISD::SETOGT,      MVT::f32, Legal);
      setCondCodeAction(ISD::SETOGT,      MVT::f64, Legal);
      setCondCodeAction(ISD::SETUGT,      MVT::f32, Legal);
      setCondCodeAction(ISD::SETUGT,      MVT::f64, Legal);

      setCondCodeAction(ISD::SETOLE,      MVT::f32, Legal);
      setCondCodeAction(ISD::SETOLE,      MVT::f64, Legal);
      setCondCodeAction(ISD::SETOLT,      MVT::f32, Legal);
      setCondCodeAction(ISD::SETOLT,      MVT::f64, Legal);

      setOperationAction(ISD::ConstantFP,  MVT::f32, Legal);
      setOperationAction(ISD::ConstantFP,  MVT::f64, Legal);

      setOperationAction(ISD::FP_TO_UINT, MVT::i1, Promote);
      setOperationAction(ISD::FP_TO_SINT, MVT::i1, Promote);
      setOperationAction(ISD::UINT_TO_FP, MVT::i1, Promote);
      setOperationAction(ISD::SINT_TO_FP, MVT::i1, Promote);

      setOperationAction(ISD::FP_TO_UINT, MVT::i8, Promote);
      setOperationAction(ISD::FP_TO_SINT, MVT::i8, Promote);
      setOperationAction(ISD::UINT_TO_FP, MVT::i8, Promote);
      setOperationAction(ISD::SINT_TO_FP, MVT::i8, Promote);

      setOperationAction(ISD::FP_TO_UINT, MVT::i16, Promote);
      setOperationAction(ISD::FP_TO_SINT, MVT::i16, Promote);
      setOperationAction(ISD::UINT_TO_FP, MVT::i16, Promote);
      setOperationAction(ISD::SINT_TO_FP, MVT::i16, Promote);

      setOperationAction(ISD::FP_TO_UINT, MVT::i32, Legal);
      setOperationAction(ISD::FP_TO_SINT, MVT::i32, Legal);
      setOperationAction(ISD::UINT_TO_FP, MVT::i32, Legal);
      setOperationAction(ISD::SINT_TO_FP, MVT::i32, Legal);

      setOperationAction(ISD::FP_TO_UINT, MVT::i64, Legal);
      setOperationAction(ISD::FP_TO_SINT, MVT::i64, Legal);
      setOperationAction(ISD::UINT_TO_FP, MVT::i64, Legal);
      setOperationAction(ISD::SINT_TO_FP, MVT::i64, Legal);

      setOperationAction(ISD::FABS,  MVT::f32, Legal);
      setOperationAction(ISD::FABS,  MVT::f64, Expand);

      setOperationAction(ISD::FNEG,  MVT::f32, Legal);
      setOperationAction(ISD::FNEG,  MVT::f64, Expand);
    } else {

      // Expand fp<->uint.
      setOperationAction(ISD::FP_TO_SINT,  MVT::i32, Expand);
      setOperationAction(ISD::FP_TO_UINT,  MVT::i32, Expand);

      setOperationAction(ISD::SINT_TO_FP,  MVT::i32, Expand);
      setOperationAction(ISD::UINT_TO_FP,  MVT::i32, Expand);

      setLibcallName(RTLIB::SINTTOFP_I64_F32, "__hexagon_floatdisf");
      setLibcallName(RTLIB::UINTTOFP_I64_F32, "__hexagon_floatundisf");

      setLibcallName(RTLIB::UINTTOFP_I32_F32, "__hexagon_floatunsisf");
      setLibcallName(RTLIB::SINTTOFP_I32_F32, "__hexagon_floatsisf");

      setLibcallName(RTLIB::SINTTOFP_I64_F64, "__hexagon_floatdidf");
      setLibcallName(RTLIB::UINTTOFP_I64_F64, "__hexagon_floatundidf");

      setLibcallName(RTLIB::UINTTOFP_I32_F64, "__hexagon_floatunsidf");
      setLibcallName(RTLIB::SINTTOFP_I32_F64, "__hexagon_floatsidf");

      setLibcallName(RTLIB::FPTOUINT_F32_I32, "__hexagon_fixunssfsi");
      setLibcallName(RTLIB::FPTOUINT_F32_I64, "__hexagon_fixunssfdi");

      setLibcallName(RTLIB::FPTOSINT_F64_I64, "__hexagon_fixdfdi");
      setLibcallName(RTLIB::FPTOSINT_F32_I64, "__hexagon_fixsfdi");

      setLibcallName(RTLIB::FPTOUINT_F64_I32, "__hexagon_fixunsdfsi");
      setLibcallName(RTLIB::FPTOUINT_F64_I64, "__hexagon_fixunsdfdi");

      setLibcallName(RTLIB::ADD_F64, "__hexagon_adddf3");
      setOperationAction(ISD::FADD,  MVT::f64, Expand);

      setLibcallName(RTLIB::ADD_F32, "__hexagon_addsf3");
      setOperationAction(ISD::FADD,  MVT::f32, Expand);

      setLibcallName(RTLIB::FPEXT_F32_F64, "__hexagon_extendsfdf2");
      setOperationAction(ISD::FP_EXTEND,  MVT::f32, Expand);

      setLibcallName(RTLIB::OEQ_F32, "__hexagon_eqsf2");
      setCondCodeAction(ISD::SETOEQ, MVT::f32, Expand);

      setLibcallName(RTLIB::OEQ_F64, "__hexagon_eqdf2");
      setCondCodeAction(ISD::SETOEQ, MVT::f64, Expand);

      setLibcallName(RTLIB::OGE_F32, "__hexagon_gesf2");
      setCondCodeAction(ISD::SETOGE, MVT::f32, Expand);

      setLibcallName(RTLIB::OGE_F64, "__hexagon_gedf2");
      setCondCodeAction(ISD::SETOGE, MVT::f64, Expand);

      setLibcallName(RTLIB::OGT_F32, "__hexagon_gtsf2");
      setCondCodeAction(ISD::SETOGT, MVT::f32, Expand);

      setLibcallName(RTLIB::OGT_F64, "__hexagon_gtdf2");
      setCondCodeAction(ISD::SETOGT, MVT::f64, Expand);

      setLibcallName(RTLIB::FPTOSINT_F64_I32, "__hexagon_fixdfsi");
      setOperationAction(ISD::FP_TO_SINT, MVT::f64, Expand);

      setLibcallName(RTLIB::FPTOSINT_F32_I32, "__hexagon_fixsfsi");
      setOperationAction(ISD::FP_TO_SINT, MVT::f32, Expand);

      setLibcallName(RTLIB::OLE_F64, "__hexagon_ledf2");
      setCondCodeAction(ISD::SETOLE, MVT::f64, Expand);

      setLibcallName(RTLIB::OLE_F32, "__hexagon_lesf2");
      setCondCodeAction(ISD::SETOLE, MVT::f32, Expand);

      setLibcallName(RTLIB::OLT_F64, "__hexagon_ltdf2");
      setCondCodeAction(ISD::SETOLT, MVT::f64, Expand);

      setLibcallName(RTLIB::OLT_F32, "__hexagon_ltsf2");
      setCondCodeAction(ISD::SETOLT, MVT::f32, Expand);

      setLibcallName(RTLIB::MUL_F64, "__hexagon_muldf3");
      setOperationAction(ISD::FMUL, MVT::f64, Expand);

      setLibcallName(RTLIB::MUL_F32, "__hexagon_mulsf3");
      setOperationAction(ISD::MUL, MVT::f32, Expand);

      setLibcallName(RTLIB::UNE_F64, "__hexagon_nedf2");
      setCondCodeAction(ISD::SETUNE, MVT::f64, Expand);

      setLibcallName(RTLIB::UNE_F32, "__hexagon_nesf2");

      setLibcallName(RTLIB::SUB_F64, "__hexagon_subdf3");
      setOperationAction(ISD::SUB, MVT::f64, Expand);

      setLibcallName(RTLIB::SUB_F32, "__hexagon_subsf3");
      setOperationAction(ISD::SUB, MVT::f32, Expand);

      setLibcallName(RTLIB::FPROUND_F64_F32, "__hexagon_truncdfsf2");
      setOperationAction(ISD::FP_ROUND, MVT::f64, Expand);

      setLibcallName(RTLIB::UO_F64, "__hexagon_unorddf2");
      setCondCodeAction(ISD::SETUO, MVT::f64, Expand);

      setLibcallName(RTLIB::O_F64, "__hexagon_unorddf2");
      setCondCodeAction(ISD::SETO, MVT::f64, Expand);

      setLibcallName(RTLIB::O_F32, "__hexagon_unordsf2");
      setCondCodeAction(ISD::SETO, MVT::f32, Expand);

      setLibcallName(RTLIB::UO_F32, "__hexagon_unordsf2");
      setCondCodeAction(ISD::SETUO, MVT::f32, Expand);

      setOperationAction(ISD::FABS,  MVT::f32, Expand);
      setOperationAction(ISD::FABS,  MVT::f64, Expand);
      setOperationAction(ISD::FNEG,  MVT::f32, Expand);
      setOperationAction(ISD::FNEG,  MVT::f64, Expand);
    }

    setLibcallName(RTLIB::SREM_I32, "__hexagon_modsi3");
    setOperationAction(ISD::SREM, MVT::i32, Expand);

    setIndexedLoadAction(ISD::POST_INC, MVT::i8, Legal);
    setIndexedLoadAction(ISD::POST_INC, MVT::i16, Legal);
    setIndexedLoadAction(ISD::POST_INC, MVT::i32, Legal);
    setIndexedLoadAction(ISD::POST_INC, MVT::i64, Legal);

    setIndexedStoreAction(ISD::POST_INC, MVT::i8, Legal);
    setIndexedStoreAction(ISD::POST_INC, MVT::i16, Legal);
    setIndexedStoreAction(ISD::POST_INC, MVT::i32, Legal);
    setIndexedStoreAction(ISD::POST_INC, MVT::i64, Legal);

    setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand);

    // Turn FP extload into load/fextend.
    setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);
    // Hexagon has a i1 sign extending load.
    setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Expand);
    // Turn FP truncstore into trunc + store.
    setTruncStoreAction(MVT::f64, MVT::f32, Expand);

    // Custom legalize GlobalAddress nodes into CONST32.
    setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
    setOperationAction(ISD::GlobalAddress, MVT::i8, Custom);
    // Truncate action?
    setOperationAction(ISD::TRUNCATE, MVT::i64, Expand);

    // Hexagon doesn't have sext_inreg, replace them with shl/sra.
    setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1 , Expand);

    // Hexagon has no REM or DIVREM operations.
    setOperationAction(ISD::UREM, MVT::i32, Expand);
    setOperationAction(ISD::SREM, MVT::i32, Expand);
    setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
    setOperationAction(ISD::UDIVREM, MVT::i32, Expand);
    setOperationAction(ISD::SREM, MVT::i64, Expand);
    setOperationAction(ISD::SDIVREM, MVT::i64, Expand);
    setOperationAction(ISD::UDIVREM, MVT::i64, Expand);

    setOperationAction(ISD::BSWAP, MVT::i64, Expand);

    // Lower SELECT_CC to SETCC and SELECT.
    setOperationAction(ISD::SELECT_CC, MVT::i32,   Custom);
    setOperationAction(ISD::SELECT_CC, MVT::i64,   Custom);

    if (QRI->Subtarget.hasV5TOps()) {

      // We need to make the operation type of SELECT node to be Custom,
      // such that we don't go into the infinite loop of
      // select ->  setcc -> select_cc -> select loop.
      setOperationAction(ISD::SELECT, MVT::f32, Custom);
      setOperationAction(ISD::SELECT, MVT::f64, Custom);

      setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
      setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);
      setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);

    } else {

      // Hexagon has no select or setcc: expand to SELECT_CC.
      setOperationAction(ISD::SELECT, MVT::f32, Expand);
      setOperationAction(ISD::SELECT, MVT::f64, Expand);

      // This is a workaround documented in DAGCombiner.cpp:2892 We don't
      // support SELECT_CC on every type.
      setOperationAction(ISD::SELECT_CC, MVT::Other,   Expand);

    }

    setOperationAction(ISD::BR_CC, MVT::Other, Expand);
    setOperationAction(ISD::BRIND, MVT::Other, Expand);
    if (EmitJumpTables) {
      setOperationAction(ISD::BR_JT, MVT::Other, Custom);
    } else {
      setOperationAction(ISD::BR_JT, MVT::Other, Expand);
    }
    // Increase jump tables cutover to 5, was 4.
    setMinimumJumpTableEntries(5);

    setOperationAction(ISD::BR_CC, MVT::i32, Expand);

    setOperationAction(ISD::MEMBARRIER, MVT::Other, Custom);
    setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);

    setOperationAction(ISD::FSIN , MVT::f64, Expand);
    setOperationAction(ISD::FCOS , MVT::f64, Expand);
    setOperationAction(ISD::FREM , MVT::f64, Expand);
    setOperationAction(ISD::FSIN , MVT::f32, Expand);
    setOperationAction(ISD::FCOS , MVT::f32, Expand);
    setOperationAction(ISD::FREM , MVT::f32, Expand);
    setOperationAction(ISD::CTPOP, MVT::i32, Expand);
    setOperationAction(ISD::CTTZ , MVT::i32, Expand);
    setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
    setOperationAction(ISD::CTLZ , MVT::i32, Expand);
    setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);
    setOperationAction(ISD::ROTL , MVT::i32, Expand);
    setOperationAction(ISD::ROTR , MVT::i32, Expand);
    setOperationAction(ISD::BSWAP, MVT::i32, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
    setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
    setOperationAction(ISD::FPOW , MVT::f64, Expand);
    setOperationAction(ISD::FPOW , MVT::f32, Expand);

    setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
    setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
    setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

    setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
    setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);

    setOperationAction(ISD::SMUL_LOHI, MVT::i64, Expand);
    setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);

    setOperationAction(ISD::EXCEPTIONADDR, MVT::i64, Expand);
    setOperationAction(ISD::EHSELECTION,   MVT::i64, Expand);
    setOperationAction(ISD::EXCEPTIONADDR, MVT::i32, Expand);
    setOperationAction(ISD::EHSELECTION,   MVT::i32, Expand);

    setOperationAction(ISD::EH_RETURN,     MVT::Other, Expand);

    if (TM.getSubtargetImpl()->isSubtargetV2()) {
      setExceptionPointerRegister(Hexagon::R20);
      setExceptionSelectorRegister(Hexagon::R21);
    } else {
      setExceptionPointerRegister(Hexagon::R0);
      setExceptionSelectorRegister(Hexagon::R1);
    }

    // VASTART needs to be custom lowered to use the VarArgsFrameIndex.
    setOperationAction(ISD::VASTART           , MVT::Other, Custom);

    // Use the default implementation.
    setOperationAction(ISD::VAARG             , MVT::Other, Expand);
    setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
    setOperationAction(ISD::VAEND             , MVT::Other, Expand);
    setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand);
    setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);


    setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Custom);
    setOperationAction(ISD::INLINEASM         , MVT::Other, Custom);

    setMinFunctionAlignment(2);

    // Needed for DYNAMIC_STACKALLOC expansion.
    unsigned StackRegister = TM.getRegisterInfo()->getStackRegister();
    setStackPointerRegisterToSaveRestore(StackRegister);
    setSchedulingPreference(Sched::VLIW);
}


const char*
HexagonTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
    default: return 0;
    case HexagonISD::CONST32:     return "HexagonISD::CONST32";
    case HexagonISD::ADJDYNALLOC: return "HexagonISD::ADJDYNALLOC";
    case HexagonISD::CMPICC:      return "HexagonISD::CMPICC";
    case HexagonISD::CMPFCC:      return "HexagonISD::CMPFCC";
    case HexagonISD::BRICC:       return "HexagonISD::BRICC";
    case HexagonISD::BRFCC:       return "HexagonISD::BRFCC";
    case HexagonISD::SELECT_ICC:  return "HexagonISD::SELECT_ICC";
    case HexagonISD::SELECT_FCC:  return "HexagonISD::SELECT_FCC";
    case HexagonISD::Hi:          return "HexagonISD::Hi";
    case HexagonISD::Lo:          return "HexagonISD::Lo";
    case HexagonISD::FTOI:        return "HexagonISD::FTOI";
    case HexagonISD::ITOF:        return "HexagonISD::ITOF";
    case HexagonISD::CALL:        return "HexagonISD::CALL";
    case HexagonISD::RET_FLAG:    return "HexagonISD::RET_FLAG";
    case HexagonISD::BR_JT:       return "HexagonISD::BR_JT";
    case HexagonISD::TC_RETURN:   return "HexagonISD::TC_RETURN";
  }
}

bool
HexagonTargetLowering::isTruncateFree(Type *Ty1, Type *Ty2) const {
  EVT MTy1 = EVT::getEVT(Ty1);
  EVT MTy2 = EVT::getEVT(Ty2);
  if (!MTy1.isSimple() || !MTy2.isSimple()) {
    return false;
  }
  return ((MTy1.getSimpleVT() == MVT::i64) && (MTy2.getSimpleVT() == MVT::i32));
}

bool HexagonTargetLowering::isTruncateFree(EVT VT1, EVT VT2) const {
  if (!VT1.isSimple() || !VT2.isSimple()) {
    return false;
  }
  return ((VT1.getSimpleVT() == MVT::i64) && (VT2.getSimpleVT() == MVT::i32));
}

SDValue
HexagonTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
    default: llvm_unreachable("Should not custom lower this!");
    case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
      // Frame & Return address.  Currently unimplemented.
    case ISD::RETURNADDR:         return LowerRETURNADDR(Op, DAG);
    case ISD::FRAMEADDR:          return LowerFRAMEADDR(Op, DAG);
    case ISD::GlobalTLSAddress:
                          llvm_unreachable("TLS not implemented for Hexagon.");
    case ISD::MEMBARRIER:         return LowerMEMBARRIER(Op, DAG);
    case ISD::ATOMIC_FENCE:       return LowerATOMIC_FENCE(Op, DAG);
    case ISD::GlobalAddress:      return LowerGLOBALADDRESS(Op, DAG);
    case ISD::VASTART:            return LowerVASTART(Op, DAG);
    case ISD::BR_JT:              return LowerBR_JT(Op, DAG);

    case ISD::DYNAMIC_STACKALLOC: return LowerDYNAMIC_STACKALLOC(Op, DAG);
    case ISD::SELECT_CC:          return LowerSELECT_CC(Op, DAG);
    case ISD::SELECT:             return Op;
    case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
    case ISD::INLINEASM:          return LowerINLINEASM(Op, DAG);

  }
}



//===----------------------------------------------------------------------===//
//                           Hexagon Scheduler Hooks
//===----------------------------------------------------------------------===//
MachineBasicBlock *
HexagonTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                   MachineBasicBlock *BB)
const {
  switch (MI->getOpcode()) {
    case Hexagon::ADJDYNALLOC: {
      MachineFunction *MF = BB->getParent();
      HexagonMachineFunctionInfo *FuncInfo =
        MF->getInfo<HexagonMachineFunctionInfo>();
      FuncInfo->addAllocaAdjustInst(MI);
      return BB;
    }
    default: llvm_unreachable("Unexpected instr type to insert");
  } // switch
}

//===----------------------------------------------------------------------===//
// Inline Assembly Support
//===----------------------------------------------------------------------===//

std::pair<unsigned, const TargetRegisterClass*>
HexagonTargetLowering::getRegForInlineAsmConstraint(const
                                                    std::string &Constraint,
                                                    EVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':   // R0-R31
       switch (VT.getSimpleVT().SimpleTy) {
       default:
         llvm_unreachable("getRegForInlineAsmConstraint Unhandled data type");
       case MVT::i32:
       case MVT::i16:
       case MVT::i8:
       case MVT::f32:
         return std::make_pair(0U, &Hexagon::IntRegsRegClass);
       case MVT::i64:
       case MVT::f64:
         return std::make_pair(0U, &Hexagon::DoubleRegsRegClass);
      }
    default:
      llvm_unreachable("Unknown asm register class");
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

/// isFPImmLegal - Returns true if the target can instruction select the
/// specified FP immediate natively. If false, the legalizer will
/// materialize the FP immediate as a load from a constant pool.
bool HexagonTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  const HexagonRegisterInfo* QRI = TM.getRegisterInfo();
  return QRI->Subtarget.hasV5TOps();
}

/// isLegalAddressingMode - Return true if the addressing mode represented by
/// AM is legal for this target, for a load/store of the specified type.
bool HexagonTargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                                  Type *Ty) const {
  // Allows a signed-extended 11-bit immediate field.
  if (AM.BaseOffs <= -(1LL << 13) || AM.BaseOffs >= (1LL << 13)-1) {
    return false;
  }

  // No global is ever allowed as a base.
  if (AM.BaseGV) {
    return false;
  }

  int Scale = AM.Scale;
  if (Scale < 0) Scale = -Scale;
  switch (Scale) {
  case 0:  // No scale reg, "r+i", "r", or just "i".
    break;
  default: // No scaled addressing mode.
    return false;
  }
  return true;
}

/// isLegalICmpImmediate - Return true if the specified immediate is legal
/// icmp immediate, that is the target has icmp instructions which can compare
/// a register against the immediate without having to materialize the
/// immediate into a register.
bool HexagonTargetLowering::isLegalICmpImmediate(int64_t Imm) const {
  return Imm >= -512 && Imm <= 511;
}

/// IsEligibleForTailCallOptimization - Check whether the call is eligible
/// for tail call optimization. Targets which want to do tail call
/// optimization should implement this function.
bool HexagonTargetLowering::IsEligibleForTailCallOptimization(
                                 SDValue Callee,
                                 CallingConv::ID CalleeCC,
                                 bool isVarArg,
                                 bool isCalleeStructRet,
                                 bool isCallerStructRet,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 const SmallVectorImpl<ISD::InputArg> &Ins,
                                 SelectionDAG& DAG) const {
  const Function *CallerF = DAG.getMachineFunction().getFunction();
  CallingConv::ID CallerCC = CallerF->getCallingConv();
  bool CCMatch = CallerCC == CalleeCC;

  // ***************************************************************************
  //  Look for obvious safe cases to perform tail call optimization that do not
  //  require ABI changes.
  // ***************************************************************************

  // If this is a tail call via a function pointer, then don't do it!
  if (!(dyn_cast<GlobalAddressSDNode>(Callee))
      && !(dyn_cast<ExternalSymbolSDNode>(Callee))) {
    return false;
  }

  // Do not optimize if the calling conventions do not match.
  if (!CCMatch)
    return false;

  // Do not tail call optimize vararg calls.
  if (isVarArg)
    return false;

  // Also avoid tail call optimization if either caller or callee uses struct
  // return semantics.
  if (isCalleeStructRet || isCallerStructRet)
    return false;

  // In addition to the cases above, we also disable Tail Call Optimization if
  // the calling convention code that at least one outgoing argument needs to
  // go on the stack. We cannot check that here because at this point that
  // information is not available.
  return true;
}
