//===-- IA64ISelLowering.cpp - IA64 DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IA64ISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "IA64ISelLowering.h"
#include "IA64MachineFunctionInfo.h"
#include "IA64TargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
using namespace llvm;

IA64TargetLowering::IA64TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
 
  // register class for general registers
  addRegisterClass(MVT::i64, IA64::GRRegisterClass);

  // register class for FP registers
  addRegisterClass(MVT::f64, IA64::FPRegisterClass);

  // register class for predicate registers
  addRegisterClass(MVT::i1, IA64::PRRegisterClass);

  setLoadXAction(ISD::EXTLOAD          , MVT::i1   , Promote);

  setLoadXAction(ISD::ZEXTLOAD         , MVT::i1   , Promote);

  setLoadXAction(ISD::SEXTLOAD         , MVT::i1   , Promote);
  setLoadXAction(ISD::SEXTLOAD         , MVT::i8   , Expand);
  setLoadXAction(ISD::SEXTLOAD         , MVT::i16  , Expand);
  setLoadXAction(ISD::SEXTLOAD         , MVT::i32  , Expand);

  setOperationAction(ISD::BRIND            , MVT::Other, Expand);
  setOperationAction(ISD::BR_JT            , MVT::Other, Expand);
  setOperationAction(ISD::BR_CC            , MVT::Other, Expand);
  setOperationAction(ISD::FP_ROUND_INREG   , MVT::f32  , Expand);

  // ia64 uses SELECT not SELECT_CC
  setOperationAction(ISD::SELECT_CC        , MVT::Other,  Expand);
  
  // We need to handle ISD::RET for void functions ourselves,
  // so we get a chance to restore ar.pfs before adding a
  // br.ret insn
  setOperationAction(ISD::RET, MVT::Other, Custom);

  setShiftAmountType(MVT::i64);

  setOperationAction(ISD::FREM             , MVT::f32  , Expand);
  setOperationAction(ISD::FREM             , MVT::f64  , Expand);

  setOperationAction(ISD::UREM             , MVT::f32  , Expand);
  setOperationAction(ISD::UREM             , MVT::f64  , Expand);

  setOperationAction(ISD::MEMBARRIER       , MVT::Other, Expand);

  setOperationAction(ISD::SINT_TO_FP       , MVT::i1   , Promote);
  setOperationAction(ISD::UINT_TO_FP       , MVT::i1   , Promote);

  // We don't support sin/cos/sqrt/pow
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FSQRT, MVT::f64, Expand);
  setOperationAction(ISD::FPOW , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::FSQRT, MVT::f32, Expand);
  setOperationAction(ISD::FPOW , MVT::f32, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1   , Expand);
    
  // FIXME: IA64 supports fcopysign natively!
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
  
  // We don't have line number support yet.
  setOperationAction(ISD::DBG_STOPPOINT, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::DBG_LABEL, MVT::Other, Expand);
  setOperationAction(ISD::EH_LABEL, MVT::Other, Expand);

  // IA64 has ctlz in the form of the 'fnorm' instruction.  The Legalizer 
  // expansion for ctlz/cttz in terms of ctpop is much larger, but lower
  // latency.
  // FIXME: Custom lower CTLZ when compiling for size?
  setOperationAction(ISD::CTLZ , MVT::i64  , Expand);
  setOperationAction(ISD::CTTZ , MVT::i64  , Expand);
  setOperationAction(ISD::ROTL , MVT::i64  , Expand);
  setOperationAction(ISD::ROTR , MVT::i64  , Expand);

  // FIXME: IA64 has this, but is not implemented. should be mux @rev
  setOperationAction(ISD::BSWAP, MVT::i64  , Expand);

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VAARG             , MVT::Other, Custom);
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  
  // Use the default implementation.
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);

  // Thread Local Storage
  setOperationAction(ISD::GlobalTLSAddress, MVT::i64, Custom);

  setStackPointerRegisterToSaveRestore(IA64::r12);

  setJumpBufSize(704); // on ia64-linux, jmp_bufs are 704 bytes..
  setJumpBufAlignment(16); // ...and must be 16-byte aligned
  
  computeRegisterProperties();

  addLegalFPImmediate(APFloat(+0.0));
  addLegalFPImmediate(APFloat(-0.0));
  addLegalFPImmediate(APFloat(+1.0));
  addLegalFPImmediate(APFloat(-1.0));
}

const char *IA64TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case IA64ISD::GETFD:  return "IA64ISD::GETFD";
  case IA64ISD::BRCALL: return "IA64ISD::BRCALL";  
  case IA64ISD::RET_FLAG: return "IA64ISD::RET_FLAG";
  }
}
  
MVT IA64TargetLowering::getSetCCResultType(const SDValue &) const {
  return MVT::i1;
}

void IA64TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG,
                                        SmallVectorImpl<SDValue> &ArgValues) {
  //
  // add beautiful description of IA64 stack frame format
  // here (from intel 24535803.pdf most likely)
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  
  GP = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i64));
  SP = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i64));
  RP = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i64));
  
  MachineBasicBlock& BB = MF.front();

  unsigned args_int[] = {IA64::r32, IA64::r33, IA64::r34, IA64::r35,
                         IA64::r36, IA64::r37, IA64::r38, IA64::r39};

  unsigned args_FP[] = {IA64::F8, IA64::F9, IA64::F10, IA64::F11,
                        IA64::F12,IA64::F13,IA64::F14, IA64::F15};

  unsigned argVreg[8];
  unsigned argPreg[8];
  unsigned argOpc[8];

  unsigned used_FPArgs = 0; // how many FP args have been used so far?

  unsigned ArgOffset = 0;
  int count = 0;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    {
      SDValue newroot, argt;
      if(count < 8) { // need to fix this logic? maybe.

        switch (getValueType(I->getType()).getSimpleVT()) {
          default:
            assert(0 && "ERROR in LowerArgs: can't lower this type of arg.\n"); 
          case MVT::f32:
            // fixme? (well, will need to for weird FP structy stuff,
            // see intel ABI docs)
          case MVT::f64:
//XXX            BuildMI(&BB, IA64::IDEF, 0, args_FP[used_FPArgs]);
            MF.getRegInfo().addLiveIn(args_FP[used_FPArgs]);
            // mark this reg as liveIn
            // floating point args go into f8..f15 as-needed, the increment
            argVreg[count] =                              // is below..:
            MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::f64));
            // FP args go into f8..f15 as needed: (hence the ++)
            argPreg[count] = args_FP[used_FPArgs++];
            argOpc[count] = IA64::FMOV;
            argt = newroot = DAG.getCopyFromReg(DAG.getRoot(), argVreg[count],
                                                MVT::f64);
            if (I->getType() == Type::FloatTy)
              argt = DAG.getNode(ISD::FP_ROUND, MVT::f32, argt,
                                 DAG.getIntPtrConstant(0));
            break;
          case MVT::i1: // NOTE: as far as C abi stuff goes,
                        // bools are just boring old ints
          case MVT::i8:
          case MVT::i16:
          case MVT::i32:
          case MVT::i64:
//XXX            BuildMI(&BB, IA64::IDEF, 0, args_int[count]);
            MF.getRegInfo().addLiveIn(args_int[count]);
            // mark this register as liveIn
            argVreg[count] =
            MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i64));
            argPreg[count] = args_int[count];
            argOpc[count] = IA64::MOV;
            argt = newroot =
              DAG.getCopyFromReg(DAG.getRoot(), argVreg[count], MVT::i64);
            if ( getValueType(I->getType()) != MVT::i64)
              argt = DAG.getNode(ISD::TRUNCATE, getValueType(I->getType()),
                  newroot);
            break;
        }
      } else { // more than 8 args go into the frame
        // Create the frame index object for this incoming parameter...
        ArgOffset = 16 + 8 * (count - 8);
        int FI = MFI->CreateFixedObject(8, ArgOffset);

        // Create the SelectionDAG nodes corresponding to a load
        //from this parameter
        SDValue FIN = DAG.getFrameIndex(FI, MVT::i64);
        argt = newroot = DAG.getLoad(getValueType(I->getType()),
                                     DAG.getEntryNode(), FIN, NULL, 0);
      }
      ++count;
      DAG.setRoot(newroot.getValue(1));
      ArgValues.push_back(argt);
    }


  // Create a vreg to hold the output of (what will become)
  // the "alloc" instruction
  VirtGPR = MF.getRegInfo().createVirtualRegister(getRegClassFor(MVT::i64));
  BuildMI(&BB, TII->get(IA64::PSEUDO_ALLOC), VirtGPR);
  // we create a PSEUDO_ALLOC (pseudo)instruction for now
/*
  BuildMI(&BB, IA64::IDEF, 0, IA64::r1);

  // hmm:
  BuildMI(&BB, IA64::IDEF, 0, IA64::r12);
  BuildMI(&BB, IA64::IDEF, 0, IA64::rp);
  // ..hmm.
  
  BuildMI(&BB, IA64::MOV, 1, GP).addReg(IA64::r1);

  // hmm:
  BuildMI(&BB, IA64::MOV, 1, SP).addReg(IA64::r12);
  BuildMI(&BB, IA64::MOV, 1, RP).addReg(IA64::rp);
  // ..hmm.
*/

  unsigned tempOffset=0;

  // if this is a varargs function, we simply lower llvm.va_start by
  // pointing to the first entry
  if(F.isVarArg()) {
    tempOffset=0;
    VarArgsFrameIndex = MFI->CreateFixedObject(8, tempOffset);
  }

  // here we actually do the moving of args, and store them to the stack
  // too if this is a varargs function:
  for (int i = 0; i < count && i < 8; ++i) {
    BuildMI(&BB, TII->get(argOpc[i]), argVreg[i]).addReg(argPreg[i]);
    if(F.isVarArg()) {
      // if this is a varargs function, we copy the input registers to the stack
      int FI = MFI->CreateFixedObject(8, tempOffset);
      tempOffset+=8;   //XXX: is it safe to use r22 like this?
      BuildMI(&BB, TII->get(IA64::MOV), IA64::r22).addFrameIndex(FI);
      // FIXME: we should use st8.spill here, one day
      BuildMI(&BB, TII->get(IA64::ST8), IA64::r22).addReg(argPreg[i]);
    }
  }

  // Finally, inform the code generator which regs we return values in.
  // (see the ISD::RET: case in the instruction selector)
  switch (getValueType(F.getReturnType()).getSimpleVT()) {
  default: assert(0 && "i have no idea where to return this type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
    MF.getRegInfo().addLiveOut(IA64::r8);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.getRegInfo().addLiveOut(IA64::F8);
    break;
  }
}

std::pair<SDValue, SDValue>
IA64TargetLowering::LowerCallTo(SDValue Chain, const Type *RetTy,
                                bool RetSExt, bool RetZExt, bool isVarArg,
                                bool isInreg, unsigned CallingConv, 
                                bool isTailCall, SDValue Callee, 
                                ArgListTy &Args, SelectionDAG &DAG) {

  MachineFunction &MF = DAG.getMachineFunction();

  unsigned NumBytes = 16;
  unsigned outRegsUsed = 0;

  if (Args.size() > 8) {
    NumBytes += (Args.size() - 8) * 8;
    outRegsUsed = 8;
  } else {
    outRegsUsed = Args.size();
  }

  // FIXME? this WILL fail if we ever try to pass around an arg that
  // consumes more than a single output slot (a 'real' double, int128
  // some sort of aggregate etc.), as we'll underestimate how many 'outX'
  // registers we use. Hopefully, the assembler will notice.
  MF.getInfo<IA64FunctionInfo>()->outRegsUsed=
    std::max(outRegsUsed, MF.getInfo<IA64FunctionInfo>()->outRegsUsed);

  // keep stack frame 16-byte aligned
  // assert(NumBytes==((NumBytes+15) & ~15) && 
  //        "stack frame not 16-byte aligned!");
  NumBytes = (NumBytes+15) & ~15;
  
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true));

  SDValue StackPtr;
  std::vector<SDValue> Stores;
  std::vector<SDValue> Converts;
  std::vector<SDValue> RegValuesToPass;
  unsigned ArgOffset = 16;
  
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    {
      SDValue Val = Args[i].Node;
      MVT ObjectVT = Val.getValueType();
      SDValue ValToStore(0, 0), ValToConvert(0, 0);
      unsigned ObjSize=8;
      switch (ObjectVT.getSimpleVT()) {
      default: assert(0 && "unexpected argument type!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32: {
        //promote to 64-bits, sign/zero extending based on type
        //of the argument
        ISD::NodeType ExtendKind = ISD::ANY_EXTEND;
        if (Args[i].isSExt)
          ExtendKind = ISD::SIGN_EXTEND;
        else if (Args[i].isZExt)
          ExtendKind = ISD::ZERO_EXTEND;
        Val = DAG.getNode(ExtendKind, MVT::i64, Val);
        // XXX: fall through
      }
      case MVT::i64:
        //ObjSize = 8;
        if(RegValuesToPass.size() >= 8) {
          ValToStore = Val;
        } else {
          RegValuesToPass.push_back(Val);
        }
        break;
      case MVT::f32:
        //promote to 64-bits
        Val = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Val);
        // XXX: fall through
      case MVT::f64:
        if(RegValuesToPass.size() >= 8) {
          ValToStore = Val;
        } else {
          RegValuesToPass.push_back(Val);
          if(1 /* TODO: if(calling external or varadic function)*/ ) {
            ValToConvert = Val; // additionally pass this FP value as an int
          }
        }
        break;
      }
      
      if(ValToStore.getNode()) {
        if(!StackPtr.getNode()) {
          StackPtr = DAG.getRegister(IA64::r12, MVT::i64);
        }
        SDValue PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
        PtrOff = DAG.getNode(ISD::ADD, MVT::i64, StackPtr, PtrOff);
        Stores.push_back(DAG.getStore(Chain, ValToStore, PtrOff, NULL, 0));
        ArgOffset += ObjSize;
      }

      if(ValToConvert.getNode()) {
        Converts.push_back(DAG.getNode(IA64ISD::GETFD, MVT::i64, ValToConvert));
      }
    }

  // Emit all stores, make sure they occur before any copies into physregs.
  if (!Stores.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &Stores[0],Stores.size());

  static const unsigned IntArgRegs[] = {
    IA64::out0, IA64::out1, IA64::out2, IA64::out3, 
    IA64::out4, IA64::out5, IA64::out6, IA64::out7
  };

  static const unsigned FPArgRegs[] = {
    IA64::F8,  IA64::F9,  IA64::F10, IA64::F11, 
    IA64::F12, IA64::F13, IA64::F14, IA64::F15
  };

  SDValue InFlag;
  
  // save the current GP, SP and RP : FIXME: do we need to do all 3 always?
  SDValue GPBeforeCall = DAG.getCopyFromReg(Chain, IA64::r1, MVT::i64, InFlag);
  Chain = GPBeforeCall.getValue(1);
  InFlag = Chain.getValue(2);
  SDValue SPBeforeCall = DAG.getCopyFromReg(Chain, IA64::r12, MVT::i64, InFlag);
  Chain = SPBeforeCall.getValue(1);
  InFlag = Chain.getValue(2);
  SDValue RPBeforeCall = DAG.getCopyFromReg(Chain, IA64::rp, MVT::i64, InFlag);
  Chain = RPBeforeCall.getValue(1);
  InFlag = Chain.getValue(2);

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing integer args into regs out[0-7]
  // mapped 1:1 and the FP args into regs F8-F15 "lazily"
  // TODO: for performance, we should only copy FP args into int regs when we
  // know this is required (i.e. for varardic or external (unknown) functions)

  // first to the FP->(integer representation) conversions, these are
  // flagged for now, but shouldn't have to be (TODO)
  unsigned seenConverts = 0;
  for (unsigned i = 0, e = RegValuesToPass.size(); i != e; ++i) {
    if(RegValuesToPass[i].getValueType().isFloatingPoint()) {
      Chain = DAG.getCopyToReg(Chain, IntArgRegs[i], Converts[seenConverts++], 
                               InFlag);
      InFlag = Chain.getValue(1);
    }
  }

  // next copy args into the usual places, these are flagged
  unsigned usedFPArgs = 0;
  for (unsigned i = 0, e = RegValuesToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain,
      RegValuesToPass[i].getValueType().isInteger() ?
        IntArgRegs[i] : FPArgRegs[usedFPArgs++], RegValuesToPass[i], InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
/*
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i64);
  }
*/

  std::vector<MVT> NodeTys;
  std::vector<SDValue> CallOperands;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  CallOperands.push_back(Chain);
  CallOperands.push_back(Callee);

  // emit the call itself
  if (InFlag.getNode())
    CallOperands.push_back(InFlag);
  else
    assert(0 && "this should never happen!\n");

  // to make way for a hack:
  Chain = DAG.getNode(IA64ISD::BRCALL, NodeTys,
                      &CallOperands[0], CallOperands.size());
  InFlag = Chain.getValue(1);

  // restore the GP, SP and RP after the call  
  Chain = DAG.getCopyToReg(Chain, IA64::r1, GPBeforeCall, InFlag);
  InFlag = Chain.getValue(1);
  Chain = DAG.getCopyToReg(Chain, IA64::r12, SPBeforeCall, InFlag);
  InFlag = Chain.getValue(1);
  Chain = DAG.getCopyToReg(Chain, IA64::rp, RPBeforeCall, InFlag);
  InFlag = Chain.getValue(1);
 
  std::vector<MVT> RetVals;
  RetVals.push_back(MVT::Other);
  RetVals.push_back(MVT::Flag);
 
  MVT RetTyVT = getValueType(RetTy);
  SDValue RetVal;
  if (RetTyVT != MVT::isVoid) {
    switch (RetTyVT.getSimpleVT()) {
    default: assert(0 && "Unknown value type to return!");
    case MVT::i1: { // bools are just like other integers (returned in r8)
      // we *could* fall through to the truncate below, but this saves a
      // few redundant predicate ops
      SDValue boolInR8 = DAG.getCopyFromReg(Chain, IA64::r8, MVT::i64,InFlag);
      InFlag = boolInR8.getValue(2);
      Chain = boolInR8.getValue(1);
      SDValue zeroReg = DAG.getCopyFromReg(Chain, IA64::r0, MVT::i64, InFlag);
      InFlag = zeroReg.getValue(2);
      Chain = zeroReg.getValue(1);
      
      RetVal = DAG.getSetCC(MVT::i1, boolInR8, zeroReg, ISD::SETNE);
      break;
    }
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      RetVal = DAG.getCopyFromReg(Chain, IA64::r8, MVT::i64, InFlag);
      Chain = RetVal.getValue(1);
      
      // keep track of whether it is sign or zero extended (todo: bools?)
/* XXX
      RetVal = DAG.getNode(RetTy->isSigned() ? ISD::AssertSext :ISD::AssertZext,
                           MVT::i64, RetVal, DAG.getValueType(RetTyVT));
*/
      RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
      break;
    case MVT::i64:
      RetVal = DAG.getCopyFromReg(Chain, IA64::r8, MVT::i64, InFlag);
      Chain = RetVal.getValue(1);
      InFlag = RetVal.getValue(2); // XXX dead
      break;
    case MVT::f32:
      RetVal = DAG.getCopyFromReg(Chain, IA64::F8, MVT::f64, InFlag);
      Chain = RetVal.getValue(1);
      RetVal = DAG.getNode(ISD::FP_ROUND, MVT::f32, RetVal,
                           DAG.getIntPtrConstant(0));
      break;
    case MVT::f64:
      RetVal = DAG.getCopyFromReg(Chain, IA64::F8, MVT::f64, InFlag);
      Chain = RetVal.getValue(1);
      InFlag = RetVal.getValue(2); // XXX dead
      break;
    }
  }
  
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                             DAG.getIntPtrConstant(0, true), SDValue());
  return std::make_pair(RetVal, Chain);
}

SDValue IA64TargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Should not custom lower this!");
  case ISD::GlobalTLSAddress:
    assert(0 && "TLS not implemented for IA64.");
  case ISD::RET: {
    SDValue AR_PFSVal, Copy;
    
    switch(Op.getNumOperands()) {
     default:
      assert(0 && "Do not know how to return this many arguments!");
      abort();
    case 1: 
      AR_PFSVal = DAG.getCopyFromReg(Op.getOperand(0), VirtGPR, MVT::i64);
      AR_PFSVal = DAG.getCopyToReg(AR_PFSVal.getValue(1), IA64::AR_PFS, 
                                   AR_PFSVal);
      return DAG.getNode(IA64ISD::RET_FLAG, MVT::Other, AR_PFSVal);
    case 3: {
      // Copy the result into the output register & restore ar.pfs
      MVT ArgVT = Op.getOperand(1).getValueType();
      unsigned ArgReg = ArgVT.isInteger() ? IA64::r8 : IA64::F8;

      AR_PFSVal = DAG.getCopyFromReg(Op.getOperand(0), VirtGPR, MVT::i64);
      Copy = DAG.getCopyToReg(AR_PFSVal.getValue(1), ArgReg, Op.getOperand(1),
                              SDValue());
      AR_PFSVal = DAG.getCopyToReg(Copy.getValue(0), IA64::AR_PFS, AR_PFSVal,
                                   Copy.getValue(1));
      return DAG.getNode(IA64ISD::RET_FLAG, MVT::Other,
                         AR_PFSVal, AR_PFSVal.getValue(1));
    }
    }
    return SDValue();
  }
  case ISD::VAARG: {
    MVT VT = getPointerTy();
    const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
    SDValue VAList = DAG.getLoad(VT, Op.getOperand(0), Op.getOperand(1), 
                                   SV, 0);
    // Increment the pointer, VAList, to the next vaarg
    SDValue VAIncr = DAG.getNode(ISD::ADD, VT, VAList, 
                                   DAG.getConstant(VT.getSizeInBits()/8,
                                                   VT));
    // Store the incremented VAList to the legalized pointer
    VAIncr = DAG.getStore(VAList.getValue(1), VAIncr,
                          Op.getOperand(1), SV, 0);
    // Load the actual argument out of the pointer VAList
    return DAG.getLoad(Op.getValueType(), VAIncr, VAList, NULL, 0);
  }
  case ISD::VASTART: {
    // vastart just stores the address of the VarArgsFrameIndex slot into the
    // memory location argument.
    SDValue FR = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i64);
    const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
    return DAG.getStore(Op.getOperand(0), FR, Op.getOperand(1), SV, 0);
  }
  // Frame & Return address.  Currently unimplemented
  case ISD::RETURNADDR:         break;
  case ISD::FRAMEADDR:          break;
  }
  return SDValue();
}
