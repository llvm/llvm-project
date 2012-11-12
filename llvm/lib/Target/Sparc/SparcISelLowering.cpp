//===-- SparcISelLowering.cpp - Sparc DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interfaces that Sparc uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "SparcISelLowering.h"
#include "SparcTargetMachine.h"
#include "SparcMachineFunctionInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;


//===----------------------------------------------------------------------===//
// Calling Convention Implementation
//===----------------------------------------------------------------------===//

static bool CC_Sparc_Assign_SRet(unsigned &ValNo, MVT &ValVT,
                                 MVT &LocVT, CCValAssign::LocInfo &LocInfo,
                                 ISD::ArgFlagsTy &ArgFlags, CCState &State)
{
  assert (ArgFlags.isSRet());

  //Assign SRet argument
  State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                         0,
                                         LocVT, LocInfo));
  return true;
}

static bool CC_Sparc_Assign_f64(unsigned &ValNo, MVT &ValVT,
                                MVT &LocVT, CCValAssign::LocInfo &LocInfo,
                                ISD::ArgFlagsTy &ArgFlags, CCState &State)
{
  static const uint16_t RegList[] = {
    SP::I0, SP::I1, SP::I2, SP::I3, SP::I4, SP::I5
  };
  //Try to get first reg
  if (unsigned Reg = State.AllocateReg(RegList, 6)) {
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  } else {
    //Assign whole thing in stack
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                           State.AllocateStack(8,4),
                                           LocVT, LocInfo));
    return true;
  }

  //Try to get second reg
  if (unsigned Reg = State.AllocateReg(RegList, 6))
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  else
    State.addLoc(CCValAssign::getCustomMem(ValNo, ValVT,
                                           State.AllocateStack(4,4),
                                           LocVT, LocInfo));
  return true;
}

#include "SparcGenCallingConv.inc"

SDValue
SparcTargetLowering::LowerReturn(SDValue Chain,
                                 CallingConv::ID CallConv, bool isVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 DebugLoc dl, SelectionDAG &DAG) const {

  MachineFunction &MF = DAG.getMachineFunction();

  // CCValAssign - represent the assignment of the return value to locations.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 DAG.getTarget(), RVLocs, *DAG.getContext());

  // Analize return values.
  CCInfo.AnalyzeReturn(Outs, RetCC_Sparc32);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (MF.getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        MF.getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(),
                             OutVals[i], Flag);

    // Guarantee that all emitted copies are stuck together with flags.
    Flag = Chain.getValue(1);
  }

  unsigned RetAddrOffset = 8; //Call Inst + Delay Slot
  // If the function returns a struct, copy the SRetReturnReg to I0
  if (MF.getFunction()->hasStructRetAttr()) {
    SparcMachineFunctionInfo *SFI = MF.getInfo<SparcMachineFunctionInfo>();
    unsigned Reg = SFI->getSRetReturnReg();
    if (!Reg)
      llvm_unreachable("sret virtual register not created in the entry block");
    SDValue Val = DAG.getCopyFromReg(Chain, dl, Reg, getPointerTy());
    Chain = DAG.getCopyToReg(Chain, dl, SP::I0, Val, Flag);
    Flag = Chain.getValue(1);
    if (MF.getRegInfo().liveout_empty())
      MF.getRegInfo().addLiveOut(SP::I0);
    RetAddrOffset = 12; // CallInst + Delay Slot + Unimp
  }

  SDValue RetAddrOffsetNode = DAG.getConstant(RetAddrOffset, MVT::i32);

  if (Flag.getNode())
    return DAG.getNode(SPISD::RET_FLAG, dl, MVT::Other, Chain,
                       RetAddrOffsetNode, Flag);
  return DAG.getNode(SPISD::RET_FLAG, dl, MVT::Other, Chain,
                     RetAddrOffsetNode);
}

/// LowerFormalArguments - V8 uses a very simple ABI, where all values are
/// passed in either one or two GPRs, including FP values.  TODO: we should
/// pass FP values in FP registers for fastcc functions.
SDValue
SparcTargetLowering::LowerFormalArguments(SDValue Chain,
                                          CallingConv::ID CallConv, bool isVarArg,
                                          const SmallVectorImpl<ISD::InputArg>
                                            &Ins,
                                          DebugLoc dl, SelectionDAG &DAG,
                                          SmallVectorImpl<SDValue> &InVals)
                                            const {

  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  SparcMachineFunctionInfo *FuncInfo = MF.getInfo<SparcMachineFunctionInfo>();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_Sparc32);

  const unsigned StackOffset = 92;

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    if (i == 0  && Ins[i].Flags.isSRet()) {
      //Get SRet from [%fp+64]
      int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, 64, true);
      SDValue FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
      SDValue Arg = DAG.getLoad(MVT::i32, dl, Chain, FIPtr,
                                MachinePointerInfo(),
                                false, false, false, 0);
      InVals.push_back(Arg);
      continue;
    }

    if (VA.isRegLoc()) {
      if (VA.needsCustom()) {
        assert(VA.getLocVT() == MVT::f64);
        unsigned VRegHi = RegInfo.createVirtualRegister(&SP::IntRegsRegClass);
        MF.getRegInfo().addLiveIn(VA.getLocReg(), VRegHi);
        SDValue HiVal = DAG.getCopyFromReg(Chain, dl, VRegHi, MVT::i32);

        assert(i+1 < e);
        CCValAssign &NextVA = ArgLocs[++i];

        SDValue LoVal;
        if (NextVA.isMemLoc()) {
          int FrameIdx = MF.getFrameInfo()->
            CreateFixedObject(4, StackOffset+NextVA.getLocMemOffset(),true);
          SDValue FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);
          LoVal = DAG.getLoad(MVT::i32, dl, Chain, FIPtr,
                              MachinePointerInfo(),
                              false, false, false, 0);
        } else {
          unsigned loReg = MF.addLiveIn(NextVA.getLocReg(),
                                        &SP::IntRegsRegClass);
          LoVal = DAG.getCopyFromReg(Chain, dl, loReg, MVT::i32);
        }
        SDValue WholeValue =
          DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, LoVal, HiVal);
        WholeValue = DAG.getNode(ISD::BITCAST, dl, MVT::f64, WholeValue);
        InVals.push_back(WholeValue);
        continue;
      }
      unsigned VReg = RegInfo.createVirtualRegister(&SP::IntRegsRegClass);
      MF.getRegInfo().addLiveIn(VA.getLocReg(), VReg);
      SDValue Arg = DAG.getCopyFromReg(Chain, dl, VReg, MVT::i32);
      if (VA.getLocVT() == MVT::f32)
        Arg = DAG.getNode(ISD::BITCAST, dl, MVT::f32, Arg);
      else if (VA.getLocVT() != MVT::i32) {
        Arg = DAG.getNode(ISD::AssertSext, dl, MVT::i32, Arg,
                          DAG.getValueType(VA.getLocVT()));
        Arg = DAG.getNode(ISD::TRUNCATE, dl, VA.getLocVT(), Arg);
      }
      InVals.push_back(Arg);
      continue;
    }

    assert(VA.isMemLoc());

    unsigned Offset = VA.getLocMemOffset()+StackOffset;

    if (VA.needsCustom()) {
      assert(VA.getValVT() == MVT::f64);
      //If it is double-word aligned, just load.
      if (Offset % 8 == 0) {
        int FI = MF.getFrameInfo()->CreateFixedObject(8,
                                                      Offset,
                                                      true);
        SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy());
        SDValue Load = DAG.getLoad(VA.getValVT(), dl, Chain, FIPtr,
                                   MachinePointerInfo(),
                                   false,false, false, 0);
        InVals.push_back(Load);
        continue;
      }

      int FI = MF.getFrameInfo()->CreateFixedObject(4,
                                                    Offset,
                                                    true);
      SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy());
      SDValue HiVal = DAG.getLoad(MVT::i32, dl, Chain, FIPtr,
                                  MachinePointerInfo(),
                                  false, false, false, 0);
      int FI2 = MF.getFrameInfo()->CreateFixedObject(4,
                                                     Offset+4,
                                                     true);
      SDValue FIPtr2 = DAG.getFrameIndex(FI2, getPointerTy());

      SDValue LoVal = DAG.getLoad(MVT::i32, dl, Chain, FIPtr2,
                                  MachinePointerInfo(),
                                  false, false, false, 0);

      SDValue WholeValue =
        DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, LoVal, HiVal);
      WholeValue = DAG.getNode(ISD::BITCAST, dl, MVT::f64, WholeValue);
      InVals.push_back(WholeValue);
      continue;
    }

    int FI = MF.getFrameInfo()->CreateFixedObject(4,
                                                  Offset,
                                                  true);
    SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy());
    SDValue Load ;
    if (VA.getValVT() == MVT::i32 || VA.getValVT() == MVT::f32) {
      Load = DAG.getLoad(VA.getValVT(), dl, Chain, FIPtr,
                         MachinePointerInfo(),
                         false, false, false, 0);
    } else {
      ISD::LoadExtType LoadOp = ISD::SEXTLOAD;
      // Sparc is big endian, so add an offset based on the ObjectVT.
      unsigned Offset = 4-std::max(1U, VA.getValVT().getSizeInBits()/8);
      FIPtr = DAG.getNode(ISD::ADD, dl, MVT::i32, FIPtr,
                          DAG.getConstant(Offset, MVT::i32));
      Load = DAG.getExtLoad(LoadOp, dl, MVT::i32, Chain, FIPtr,
                            MachinePointerInfo(),
                            VA.getValVT(), false, false,0);
      Load = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), Load);
    }
    InVals.push_back(Load);
  }

  if (MF.getFunction()->hasStructRetAttr()) {
    //Copy the SRet Argument to SRetReturnReg
    SparcMachineFunctionInfo *SFI = MF.getInfo<SparcMachineFunctionInfo>();
    unsigned Reg = SFI->getSRetReturnReg();
    if (!Reg) {
      Reg = MF.getRegInfo().createVirtualRegister(&SP::IntRegsRegClass);
      SFI->setSRetReturnReg(Reg);
    }
    SDValue Copy = DAG.getCopyToReg(DAG.getEntryNode(), dl, Reg, InVals[0]);
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Copy, Chain);
  }

  // Store remaining ArgRegs to the stack if this is a varargs function.
  if (isVarArg) {
    static const uint16_t ArgRegs[] = {
      SP::I0, SP::I1, SP::I2, SP::I3, SP::I4, SP::I5
    };
    unsigned NumAllocated = CCInfo.getFirstUnallocated(ArgRegs, 6);
    const uint16_t *CurArgReg = ArgRegs+NumAllocated, *ArgRegEnd = ArgRegs+6;
    unsigned ArgOffset = CCInfo.getNextStackOffset();
    if (NumAllocated == 6)
      ArgOffset += StackOffset;
    else {
      assert(!ArgOffset);
      ArgOffset = 68+4*NumAllocated;
    }

    // Remember the vararg offset for the va_start implementation.
    FuncInfo->setVarArgsFrameOffset(ArgOffset);

    std::vector<SDValue> OutChains;

    for (; CurArgReg != ArgRegEnd; ++CurArgReg) {
      unsigned VReg = RegInfo.createVirtualRegister(&SP::IntRegsRegClass);
      MF.getRegInfo().addLiveIn(*CurArgReg, VReg);
      SDValue Arg = DAG.getCopyFromReg(DAG.getRoot(), dl, VReg, MVT::i32);

      int FrameIdx = MF.getFrameInfo()->CreateFixedObject(4, ArgOffset,
                                                          true);
      SDValue FIPtr = DAG.getFrameIndex(FrameIdx, MVT::i32);

      OutChains.push_back(DAG.getStore(DAG.getRoot(), dl, Arg, FIPtr,
                                       MachinePointerInfo(),
                                       false, false, 0));
      ArgOffset += 4;
    }

    if (!OutChains.empty()) {
      OutChains.push_back(Chain);
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                          &OutChains[0], OutChains.size());
    }
  }

  return Chain;
}

SDValue
SparcTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
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

  // Sparc target does not yet support tail call optimization.
  isTailCall = false;

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 DAG.getTarget(), ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeCallOperands(Outs, CC_Sparc32);

  // Get the size of the outgoing arguments stack space requirement.
  unsigned ArgsSize = CCInfo.getNextStackOffset();

  // Keep stack frames 8-byte aligned.
  ArgsSize = (ArgsSize+7) & ~7;

  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();

  //Create local copies for byval args.
  SmallVector<SDValue, 8> ByValArgs;
  for (unsigned i = 0,  e = Outs.size(); i != e; ++i) {
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    if (!Flags.isByVal())
      continue;

    SDValue Arg = OutVals[i];
    unsigned Size = Flags.getByValSize();
    unsigned Align = Flags.getByValAlign();

    int FI = MFI->CreateStackObject(Size, Align, false);
    SDValue FIPtr = DAG.getFrameIndex(FI, getPointerTy());
    SDValue SizeNode = DAG.getConstant(Size, MVT::i32);

    Chain = DAG.getMemcpy(Chain, dl, FIPtr, Arg, SizeNode, Align,
                          false,        //isVolatile,
                          (Size <= 32), //AlwaysInline if size <= 32
                          MachinePointerInfo(), MachinePointerInfo());
    ByValArgs.push_back(FIPtr);
  }

  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(ArgsSize, true));

  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  const unsigned StackOffset = 92;
  bool hasStructRetAttr = false;
  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, realArgIdx = 0, byvalArgIdx = 0, e = ArgLocs.size();
       i != e;
       ++i, ++realArgIdx) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[realArgIdx];

    ISD::ArgFlagsTy Flags = Outs[realArgIdx].Flags;

    //Use local copy if it is a byval arg.
    if (Flags.isByVal())
      Arg = ByValArgs[byvalArgIdx++];

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
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, dl, VA.getLocVT(), Arg);
      break;
    }

    if (Flags.isSRet()) {
      assert(VA.needsCustom());
      // store SRet argument in %sp+64
      SDValue StackPtr = DAG.getRegister(SP::O6, MVT::i32);
      SDValue PtrOff = DAG.getIntPtrConstant(64);
      PtrOff = DAG.getNode(ISD::ADD, dl, MVT::i32, StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                         MachinePointerInfo(),
                                         false, false, 0));
      hasStructRetAttr = true;
      continue;
    }

    if (VA.needsCustom()) {
      assert(VA.getLocVT() == MVT::f64);

      if (VA.isMemLoc()) {
        unsigned Offset = VA.getLocMemOffset() + StackOffset;
        //if it is double-word aligned, just store.
        if (Offset % 8 == 0) {
          SDValue StackPtr = DAG.getRegister(SP::O6, MVT::i32);
          SDValue PtrOff = DAG.getIntPtrConstant(Offset);
          PtrOff = DAG.getNode(ISD::ADD, dl, MVT::i32, StackPtr, PtrOff);
          MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                             MachinePointerInfo(),
                                             false, false, 0));
          continue;
        }
      }

      SDValue StackPtr = DAG.CreateStackTemporary(MVT::f64, MVT::i32);
      SDValue Store = DAG.getStore(DAG.getEntryNode(), dl,
                                   Arg, StackPtr, MachinePointerInfo(),
                                   false, false, 0);
      // Sparc is big-endian, so the high part comes first.
      SDValue Hi = DAG.getLoad(MVT::i32, dl, Store, StackPtr,
                               MachinePointerInfo(), false, false, false, 0);
      // Increment the pointer to the other half.
      StackPtr = DAG.getNode(ISD::ADD, dl, StackPtr.getValueType(), StackPtr,
                             DAG.getIntPtrConstant(4));
      // Load the low part.
      SDValue Lo = DAG.getLoad(MVT::i32, dl, Store, StackPtr,
                               MachinePointerInfo(), false, false, false, 0);

      if (VA.isRegLoc()) {
        RegsToPass.push_back(std::make_pair(VA.getLocReg(), Hi));
        assert(i+1 != e);
        CCValAssign &NextVA = ArgLocs[++i];
        if (NextVA.isRegLoc()) {
          RegsToPass.push_back(std::make_pair(NextVA.getLocReg(), Lo));
        } else {
          //Store the low part in stack.
          unsigned Offset = NextVA.getLocMemOffset() + StackOffset;
          SDValue StackPtr = DAG.getRegister(SP::O6, MVT::i32);
          SDValue PtrOff = DAG.getIntPtrConstant(Offset);
          PtrOff = DAG.getNode(ISD::ADD, dl, MVT::i32, StackPtr, PtrOff);
          MemOpChains.push_back(DAG.getStore(Chain, dl, Lo, PtrOff,
                                             MachinePointerInfo(),
                                             false, false, 0));
        }
      } else {
        unsigned Offset = VA.getLocMemOffset() + StackOffset;
        // Store the high part.
        SDValue StackPtr = DAG.getRegister(SP::O6, MVT::i32);
        SDValue PtrOff = DAG.getIntPtrConstant(Offset);
        PtrOff = DAG.getNode(ISD::ADD, dl, MVT::i32, StackPtr, PtrOff);
        MemOpChains.push_back(DAG.getStore(Chain, dl, Hi, PtrOff,
                                           MachinePointerInfo(),
                                           false, false, 0));
        // Store the low part.
        PtrOff = DAG.getIntPtrConstant(Offset+4);
        PtrOff = DAG.getNode(ISD::ADD, dl, MVT::i32, StackPtr, PtrOff);
        MemOpChains.push_back(DAG.getStore(Chain, dl, Lo, PtrOff,
                                           MachinePointerInfo(),
                                           false, false, 0));
      }
      continue;
    }

    // Arguments that can be passed on register must be kept at
    // RegsToPass vector
    if (VA.isRegLoc()) {
      if (VA.getLocVT() != MVT::f32) {
        RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
        continue;
      }
      Arg = DAG.getNode(ISD::BITCAST, dl, MVT::i32, Arg);
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      continue;
    }

    assert(VA.isMemLoc());

    // Create a store off the stack pointer for this argument.
    SDValue StackPtr = DAG.getRegister(SP::O6, MVT::i32);
    SDValue PtrOff = DAG.getIntPtrConstant(VA.getLocMemOffset()+StackOffset);
    PtrOff = DAG.getNode(ISD::ADD, dl, MVT::i32, StackPtr, PtrOff);
    MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                       MachinePointerInfo(),
                                       false, false, 0));
  }


  // Emit all stores, make sure the occur before any copies into physregs.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emitted instructions must be
  // stuck together.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    unsigned Reg = RegsToPass[i].first;
    // Remap I0->I7 -> O0->O7.
    if (Reg >= SP::I0 && Reg <= SP::I7)
      Reg = Reg-SP::I0+SP::O0;

    Chain = DAG.getCopyToReg(Chain, dl, Reg, RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  unsigned SRetArgSize = (hasStructRetAttr)? getSRetArgSize(DAG, Callee):0;

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl, MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  // Returns a chain & a flag for retval copy to use
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  if (hasStructRetAttr)
    Ops.push_back(DAG.getTargetConstant(SRetArgSize, MVT::i32));
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    unsigned Reg = RegsToPass[i].first;
    if (Reg >= SP::I0 && Reg <= SP::I7)
      Reg = Reg-SP::I0+SP::O0;

    Ops.push_back(DAG.getRegister(Reg, RegsToPass[i].second.getValueType()));
  }
  if (InFlag.getNode())
    Ops.push_back(InFlag);

  Chain = DAG.getNode(SPISD::CALL, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(ArgsSize, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  InFlag = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState RVInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 DAG.getTarget(), RVLocs, *DAG.getContext());

  RVInfo.AnalyzeCallResult(Ins, RetCC_Sparc32);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    unsigned Reg = RVLocs[i].getLocReg();

    // Remap I0->I7 -> O0->O7.
    if (Reg >= SP::I0 && Reg <= SP::I7)
      Reg = Reg-SP::I0+SP::O0;

    Chain = DAG.getCopyFromReg(Chain, dl, Reg,
                               RVLocs[i].getValVT(), InFlag).getValue(1);
    InFlag = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

unsigned
SparcTargetLowering::getSRetArgSize(SelectionDAG &DAG, SDValue Callee) const
{
  const Function *CalleeFn = 0;
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    CalleeFn = dyn_cast<Function>(G->getGlobal());
  } else if (ExternalSymbolSDNode *E =
             dyn_cast<ExternalSymbolSDNode>(Callee)) {
    const Function *Fn = DAG.getMachineFunction().getFunction();
    const Module *M = Fn->getParent();
    CalleeFn = M->getFunction(E->getSymbol());
  }

  if (!CalleeFn)
    return 0;

  assert(CalleeFn->hasStructRetAttr() &&
         "Callee does not have the StructRet attribute.");

  PointerType *Ty = cast<PointerType>(CalleeFn->arg_begin()->getType());
  Type *ElementTy = Ty->getElementType();
  return getDataLayout()->getTypeAllocSize(ElementTy);
}

//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

/// IntCondCCodeToICC - Convert a DAG integer condition code to a SPARC ICC
/// condition.
static SPCC::CondCodes IntCondCCodeToICC(ISD::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Unknown integer condition code!");
  case ISD::SETEQ:  return SPCC::ICC_E;
  case ISD::SETNE:  return SPCC::ICC_NE;
  case ISD::SETLT:  return SPCC::ICC_L;
  case ISD::SETGT:  return SPCC::ICC_G;
  case ISD::SETLE:  return SPCC::ICC_LE;
  case ISD::SETGE:  return SPCC::ICC_GE;
  case ISD::SETULT: return SPCC::ICC_CS;
  case ISD::SETULE: return SPCC::ICC_LEU;
  case ISD::SETUGT: return SPCC::ICC_GU;
  case ISD::SETUGE: return SPCC::ICC_CC;
  }
}

/// FPCondCCodeToFCC - Convert a DAG floatingp oint condition code to a SPARC
/// FCC condition.
static SPCC::CondCodes FPCondCCodeToFCC(ISD::CondCode CC) {
  switch (CC) {
  default: llvm_unreachable("Unknown fp condition code!");
  case ISD::SETEQ:
  case ISD::SETOEQ: return SPCC::FCC_E;
  case ISD::SETNE:
  case ISD::SETUNE: return SPCC::FCC_NE;
  case ISD::SETLT:
  case ISD::SETOLT: return SPCC::FCC_L;
  case ISD::SETGT:
  case ISD::SETOGT: return SPCC::FCC_G;
  case ISD::SETLE:
  case ISD::SETOLE: return SPCC::FCC_LE;
  case ISD::SETGE:
  case ISD::SETOGE: return SPCC::FCC_GE;
  case ISD::SETULT: return SPCC::FCC_UL;
  case ISD::SETULE: return SPCC::FCC_ULE;
  case ISD::SETUGT: return SPCC::FCC_UG;
  case ISD::SETUGE: return SPCC::FCC_UGE;
  case ISD::SETUO:  return SPCC::FCC_U;
  case ISD::SETO:   return SPCC::FCC_O;
  case ISD::SETONE: return SPCC::FCC_LG;
  case ISD::SETUEQ: return SPCC::FCC_UE;
  }
}

SparcTargetLowering::SparcTargetLowering(TargetMachine &TM)
  : TargetLowering(TM, new TargetLoweringObjectFileELF()) {

  // Set up the register classes.
  addRegisterClass(MVT::i32, &SP::IntRegsRegClass);
  addRegisterClass(MVT::f32, &SP::FPRegsRegClass);
  addRegisterClass(MVT::f64, &SP::DFPRegsRegClass);

  // Turn FP extload into load/fextend
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);
  // Sparc doesn't have i1 sign extending load
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  // Turn FP truncstore into trunc + store.
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // Custom legalize GlobalAddress nodes into LO/HI parts.
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool , MVT::i32, Custom);

  // Sparc doesn't have sext_inreg, replace them with shl/sra
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8 , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1 , Expand);

  // Sparc has no REM or DIVREM operations.
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  // Custom expand fp<->sint
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);

  // Expand fp<->uint
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);

  setOperationAction(ISD::BITCAST, MVT::f32, Expand);
  setOperationAction(ISD::BITCAST, MVT::i32, Expand);

  // Sparc has no select or setcc: expand to SELECT_CC.
  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT, MVT::f32, Expand);
  setOperationAction(ISD::SELECT, MVT::f64, Expand);
  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SETCC, MVT::f32, Expand);
  setOperationAction(ISD::SETCC, MVT::f64, Expand);

  // Sparc doesn't have BRCOND either, it has BR_CC.
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);
  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f64, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);

  // FIXME: There are instructions available for ATOMIC_FENCE
  // on SparcV8 and later.
  setOperationAction(ISD::MEMBARRIER, MVT::Other, Expand);
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Expand);

  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FREM , MVT::f64, Expand);
  setOperationAction(ISD::FMA  , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::FREM , MVT::f32, Expand);
  setOperationAction(ISD::FMA  , MVT::f32, Expand);
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

  // FIXME: Sparc provides these multiplies, but we don't have them yet.
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);

  setOperationAction(ISD::EH_LABEL, MVT::Other, Expand);

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex.
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);
  // VAARG needs to be lowered to not do unaligned accesses for doubles.
  setOperationAction(ISD::VAARG             , MVT::Other, Custom);

  // Use the default implementation.
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Custom);

  // No debug info support yet.
  setOperationAction(ISD::EH_LABEL, MVT::Other, Expand);

  setStackPointerRegisterToSaveRestore(SP::O6);

  if (TM.getSubtarget<SparcSubtarget>().isV9())
    setOperationAction(ISD::CTPOP, MVT::i32, Legal);

  setMinFunctionAlignment(2);

  computeRegisterProperties();
}

const char *SparcTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case SPISD::CMPICC:     return "SPISD::CMPICC";
  case SPISD::CMPFCC:     return "SPISD::CMPFCC";
  case SPISD::BRICC:      return "SPISD::BRICC";
  case SPISD::BRFCC:      return "SPISD::BRFCC";
  case SPISD::SELECT_ICC: return "SPISD::SELECT_ICC";
  case SPISD::SELECT_FCC: return "SPISD::SELECT_FCC";
  case SPISD::Hi:         return "SPISD::Hi";
  case SPISD::Lo:         return "SPISD::Lo";
  case SPISD::FTOI:       return "SPISD::FTOI";
  case SPISD::ITOF:       return "SPISD::ITOF";
  case SPISD::CALL:       return "SPISD::CALL";
  case SPISD::RET_FLAG:   return "SPISD::RET_FLAG";
  case SPISD::GLOBAL_BASE_REG: return "SPISD::GLOBAL_BASE_REG";
  case SPISD::FLUSHW:     return "SPISD::FLUSHW";
  }
}

/// isMaskedValueZeroForTargetNode - Return true if 'Op & Mask' is known to
/// be zero. Op is expected to be a target specific node. Used by DAG
/// combiner.
void SparcTargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                         APInt &KnownZero,
                                                         APInt &KnownOne,
                                                         const SelectionDAG &DAG,
                                                         unsigned Depth) const {
  APInt KnownZero2, KnownOne2;
  KnownZero = KnownOne = APInt(KnownZero.getBitWidth(), 0);

  switch (Op.getOpcode()) {
  default: break;
  case SPISD::SELECT_ICC:
  case SPISD::SELECT_FCC:
    DAG.ComputeMaskedBits(Op.getOperand(1), KnownZero, KnownOne, Depth+1);
    DAG.ComputeMaskedBits(Op.getOperand(0), KnownZero2, KnownOne2, Depth+1);
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  }
}

// Look at LHS/RHS/CC and see if they are a lowered setcc instruction.  If so
// set LHS/RHS and SPCC to the LHS/RHS of the setcc and SPCC to the condition.
static void LookThroughSetCC(SDValue &LHS, SDValue &RHS,
                             ISD::CondCode CC, unsigned &SPCC) {
  if (isa<ConstantSDNode>(RHS) &&
      cast<ConstantSDNode>(RHS)->isNullValue() &&
      CC == ISD::SETNE &&
      ((LHS.getOpcode() == SPISD::SELECT_ICC &&
        LHS.getOperand(3).getOpcode() == SPISD::CMPICC) ||
       (LHS.getOpcode() == SPISD::SELECT_FCC &&
        LHS.getOperand(3).getOpcode() == SPISD::CMPFCC)) &&
      isa<ConstantSDNode>(LHS.getOperand(0)) &&
      isa<ConstantSDNode>(LHS.getOperand(1)) &&
      cast<ConstantSDNode>(LHS.getOperand(0))->isOne() &&
      cast<ConstantSDNode>(LHS.getOperand(1))->isNullValue()) {
    SDValue CMPCC = LHS.getOperand(3);
    SPCC = cast<ConstantSDNode>(LHS.getOperand(2))->getZExtValue();
    LHS = CMPCC.getOperand(0);
    RHS = CMPCC.getOperand(1);
  }
}

SDValue SparcTargetLowering::LowerGlobalAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  // FIXME there isn't really any debug info here
  DebugLoc dl = Op.getDebugLoc();
  SDValue GA = DAG.getTargetGlobalAddress(GV, dl, MVT::i32);
  SDValue Hi = DAG.getNode(SPISD::Hi, dl, MVT::i32, GA);
  SDValue Lo = DAG.getNode(SPISD::Lo, dl, MVT::i32, GA);

  if (getTargetMachine().getRelocationModel() != Reloc::PIC_)
    return DAG.getNode(ISD::ADD, dl, MVT::i32, Lo, Hi);

  SDValue GlobalBase = DAG.getNode(SPISD::GLOBAL_BASE_REG, dl,
                                   getPointerTy());
  SDValue RelAddr = DAG.getNode(ISD::ADD, dl, MVT::i32, Lo, Hi);
  SDValue AbsAddr = DAG.getNode(ISD::ADD, dl, MVT::i32,
                                GlobalBase, RelAddr);
  return DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(),
                     AbsAddr, MachinePointerInfo(), false, false, false, 0);
}

SDValue SparcTargetLowering::LowerConstantPool(SDValue Op,
                                               SelectionDAG &DAG) const {
  ConstantPoolSDNode *N = cast<ConstantPoolSDNode>(Op);
  // FIXME there isn't really any debug info here
  DebugLoc dl = Op.getDebugLoc();
  const Constant *C = N->getConstVal();
  SDValue CP = DAG.getTargetConstantPool(C, MVT::i32, N->getAlignment());
  SDValue Hi = DAG.getNode(SPISD::Hi, dl, MVT::i32, CP);
  SDValue Lo = DAG.getNode(SPISD::Lo, dl, MVT::i32, CP);
  if (getTargetMachine().getRelocationModel() != Reloc::PIC_)
    return DAG.getNode(ISD::ADD, dl, MVT::i32, Lo, Hi);

  SDValue GlobalBase = DAG.getNode(SPISD::GLOBAL_BASE_REG, dl,
                                   getPointerTy());
  SDValue RelAddr = DAG.getNode(ISD::ADD, dl, MVT::i32, Lo, Hi);
  SDValue AbsAddr = DAG.getNode(ISD::ADD, dl, MVT::i32,
                                GlobalBase, RelAddr);
  return DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(),
                     AbsAddr, MachinePointerInfo(), false, false, false, 0);
}

static SDValue LowerFP_TO_SINT(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  // Convert the fp value to integer in an FP register.
  assert(Op.getValueType() == MVT::i32);
  Op = DAG.getNode(SPISD::FTOI, dl, MVT::f32, Op.getOperand(0));
  return DAG.getNode(ISD::BITCAST, dl, MVT::i32, Op);
}

static SDValue LowerSINT_TO_FP(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  assert(Op.getOperand(0).getValueType() == MVT::i32);
  SDValue Tmp = DAG.getNode(ISD::BITCAST, dl, MVT::f32, Op.getOperand(0));
  // Convert the int value to FP in an FP register.
  return DAG.getNode(SPISD::ITOF, dl, Op.getValueType(), Tmp);
}

static SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  DebugLoc dl = Op.getDebugLoc();
  unsigned Opc, SPCC = ~0U;

  // If this is a br_cc of a "setcc", and if the setcc got lowered into
  // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
  LookThroughSetCC(LHS, RHS, CC, SPCC);

  // Get the condition flag.
  SDValue CompareFlag;
  if (LHS.getValueType() == MVT::i32) {
    std::vector<EVT> VTs;
    VTs.push_back(MVT::i32);
    VTs.push_back(MVT::Glue);
    SDValue Ops[2] = { LHS, RHS };
    CompareFlag = DAG.getNode(SPISD::CMPICC, dl, VTs, Ops, 2).getValue(1);
    if (SPCC == ~0U) SPCC = IntCondCCodeToICC(CC);
    Opc = SPISD::BRICC;
  } else {
    CompareFlag = DAG.getNode(SPISD::CMPFCC, dl, MVT::Glue, LHS, RHS);
    if (SPCC == ~0U) SPCC = FPCondCCodeToFCC(CC);
    Opc = SPISD::BRFCC;
  }
  return DAG.getNode(Opc, dl, MVT::Other, Chain, Dest,
                     DAG.getConstant(SPCC, MVT::i32), CompareFlag);
}

static SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);
  DebugLoc dl = Op.getDebugLoc();
  unsigned Opc, SPCC = ~0U;

  // If this is a select_cc of a "setcc", and if the setcc got lowered into
  // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
  LookThroughSetCC(LHS, RHS, CC, SPCC);

  SDValue CompareFlag;
  if (LHS.getValueType() == MVT::i32) {
    std::vector<EVT> VTs;
    VTs.push_back(LHS.getValueType());   // subcc returns a value
    VTs.push_back(MVT::Glue);
    SDValue Ops[2] = { LHS, RHS };
    CompareFlag = DAG.getNode(SPISD::CMPICC, dl, VTs, Ops, 2).getValue(1);
    Opc = SPISD::SELECT_ICC;
    if (SPCC == ~0U) SPCC = IntCondCCodeToICC(CC);
  } else {
    CompareFlag = DAG.getNode(SPISD::CMPFCC, dl, MVT::Glue, LHS, RHS);
    Opc = SPISD::SELECT_FCC;
    if (SPCC == ~0U) SPCC = FPCondCCodeToFCC(CC);
  }
  return DAG.getNode(Opc, dl, TrueVal.getValueType(), TrueVal, FalseVal,
                     DAG.getConstant(SPCC, MVT::i32), CompareFlag);
}

static SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG,
                            const SparcTargetLowering &TLI) {
  MachineFunction &MF = DAG.getMachineFunction();
  SparcMachineFunctionInfo *FuncInfo = MF.getInfo<SparcMachineFunctionInfo>();

  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  DebugLoc dl = Op.getDebugLoc();
  SDValue Offset =
    DAG.getNode(ISD::ADD, dl, MVT::i32,
                DAG.getRegister(SP::I6, MVT::i32),
                DAG.getConstant(FuncInfo->getVarArgsFrameOffset(),
                                MVT::i32));
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  return DAG.getStore(Op.getOperand(0), dl, Offset, Op.getOperand(1),
                      MachinePointerInfo(SV), false, false, 0);
}

static SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getValueType(0);
  SDValue InChain = Node->getOperand(0);
  SDValue VAListPtr = Node->getOperand(1);
  const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  DebugLoc dl = Node->getDebugLoc();
  SDValue VAList = DAG.getLoad(MVT::i32, dl, InChain, VAListPtr,
                               MachinePointerInfo(SV), false, false, false, 0);
  // Increment the pointer, VAList, to the next vaarg
  SDValue NextPtr = DAG.getNode(ISD::ADD, dl, MVT::i32, VAList,
                                  DAG.getConstant(VT.getSizeInBits()/8,
                                                  MVT::i32));
  // Store the incremented VAList to the legalized pointer
  InChain = DAG.getStore(VAList.getValue(1), dl, NextPtr,
                         VAListPtr, MachinePointerInfo(SV), false, false, 0);
  // Load the actual argument out of the pointer VAList, unless this is an
  // f64 load.
  if (VT != MVT::f64)
    return DAG.getLoad(VT, dl, InChain, VAList, MachinePointerInfo(),
                       false, false, false, 0);

  // Otherwise, load it as i64, then do a bitconvert.
  SDValue V = DAG.getLoad(MVT::i64, dl, InChain, VAList, MachinePointerInfo(),
                          false, false, false, 0);

  // Bit-Convert the value to f64.
  SDValue Ops[2] = {
    DAG.getNode(ISD::BITCAST, dl, MVT::f64, V),
    V.getValue(1)
  };
  return DAG.getMergeValues(Ops, 2, dl);
}

static SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) {
  SDValue Chain = Op.getOperand(0);  // Legalize the chain.
  SDValue Size  = Op.getOperand(1);  // Legalize the size.
  DebugLoc dl = Op.getDebugLoc();

  unsigned SPReg = SP::O6;
  SDValue SP = DAG.getCopyFromReg(Chain, dl, SPReg, MVT::i32);
  SDValue NewSP = DAG.getNode(ISD::SUB, dl, MVT::i32, SP, Size); // Value
  Chain = DAG.getCopyToReg(SP.getValue(1), dl, SPReg, NewSP);    // Output chain

  // The resultant pointer is actually 16 words from the bottom of the stack,
  // to provide a register spill area.
  SDValue NewVal = DAG.getNode(ISD::ADD, dl, MVT::i32, NewSP,
                                 DAG.getConstant(96, MVT::i32));
  SDValue Ops[2] = { NewVal, Chain };
  return DAG.getMergeValues(Ops, 2, dl);
}


static SDValue getFLUSHW(SDValue Op, SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  SDValue Chain = DAG.getNode(SPISD::FLUSHW,
                              dl, MVT::Other, DAG.getEntryNode());
  return Chain;
}

static SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setFrameAddressIsTaken(true);

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned FrameReg = SP::I6;

  uint64_t depth = Op.getConstantOperandVal(0);

  SDValue FrameAddr;
  if (depth == 0)
    FrameAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl, FrameReg, VT);
  else {
    // flush first to make sure the windowed registers' values are in stack
    SDValue Chain = getFLUSHW(Op, DAG);
    FrameAddr = DAG.getCopyFromReg(Chain, dl, FrameReg, VT);

    for (uint64_t i = 0; i != depth; ++i) {
      SDValue Ptr = DAG.getNode(ISD::ADD,
                                dl, MVT::i32,
                                FrameAddr, DAG.getIntPtrConstant(56));
      FrameAddr = DAG.getLoad(MVT::i32, dl,
                              Chain,
                              Ptr,
                              MachinePointerInfo(), false, false, false, 0);
    }
  }
  return FrameAddr;
}

static SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) {
  MachineFrameInfo *MFI = DAG.getMachineFunction().getFrameInfo();
  MFI->setReturnAddressIsTaken(true);

  EVT VT = Op.getValueType();
  DebugLoc dl = Op.getDebugLoc();
  unsigned RetReg = SP::I7;

  uint64_t depth = Op.getConstantOperandVal(0);

  SDValue RetAddr;
  if (depth == 0)
    RetAddr = DAG.getCopyFromReg(DAG.getEntryNode(), dl, RetReg, VT);
  else {
    // flush first to make sure the windowed registers' values are in stack
    SDValue Chain = getFLUSHW(Op, DAG);
    RetAddr = DAG.getCopyFromReg(Chain, dl, SP::I6, VT);

    for (uint64_t i = 0; i != depth; ++i) {
      SDValue Ptr = DAG.getNode(ISD::ADD,
                                dl, MVT::i32,
                                RetAddr,
                                DAG.getIntPtrConstant((i == depth-1)?60:56));
      RetAddr = DAG.getLoad(MVT::i32, dl,
                            Chain,
                            Ptr,
                            MachinePointerInfo(), false, false, false, 0);
    }
  }
  return RetAddr;
}

SDValue SparcTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Should not custom lower this!");
  case ISD::RETURNADDR:         return LowerRETURNADDR(Op, DAG);
  case ISD::FRAMEADDR:          return LowerFRAMEADDR(Op, DAG);
  case ISD::GlobalTLSAddress:
    llvm_unreachable("TLS not implemented for Sparc.");
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::ConstantPool:       return LowerConstantPool(Op, DAG);
  case ISD::FP_TO_SINT:         return LowerFP_TO_SINT(Op, DAG);
  case ISD::SINT_TO_FP:         return LowerSINT_TO_FP(Op, DAG);
  case ISD::BR_CC:              return LowerBR_CC(Op, DAG);
  case ISD::SELECT_CC:          return LowerSELECT_CC(Op, DAG);
  case ISD::VASTART:            return LowerVASTART(Op, DAG, *this);
  case ISD::VAARG:              return LowerVAARG(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC: return LowerDYNAMIC_STACKALLOC(Op, DAG);
  }
}

MachineBasicBlock *
SparcTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                 MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *getTargetMachine().getInstrInfo();
  unsigned BROpcode;
  unsigned CC;
  DebugLoc dl = MI->getDebugLoc();
  // Figure out the conditional branch opcode to use for this select_cc.
  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unknown SELECT_CC!");
  case SP::SELECT_CC_Int_ICC:
  case SP::SELECT_CC_FP_ICC:
  case SP::SELECT_CC_DFP_ICC:
    BROpcode = SP::BCOND;
    break;
  case SP::SELECT_CC_Int_FCC:
  case SP::SELECT_CC_FP_FCC:
  case SP::SELECT_CC_DFP_FCC:
    BROpcode = SP::FBCOND;
    break;
  }

  CC = (SPCC::CondCodes)MI->getOperand(3).getImm();

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
  //   [f]bCC copy1MBB
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

  // Add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  BuildMI(BB, dl, TII.get(BROpcode)).addMBB(sinkMBB).addImm(CC);

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
  BuildMI(*BB, BB->begin(), dl, TII.get(SP::PHI), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(thisMBB);

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return BB;
}

//===----------------------------------------------------------------------===//
//                         Sparc Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
SparcTargetLowering::ConstraintType
SparcTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:  break;
    case 'r': return C_RegisterClass;
    }
  }

  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass*>
SparcTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                  EVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, &SP::IntRegsRegClass);
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

bool
SparcTargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The Sparc target isn't yet aware of offsets.
  return false;
}
