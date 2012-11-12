//===-- ARMSelectionDAGInfo.cpp - ARM SelectionDAG Info -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARMSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-selectiondag-info"
#include "ARMTargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;

ARMSelectionDAGInfo::ARMSelectionDAGInfo(const TargetMachine &TM)
  : TargetSelectionDAGInfo(TM),
    Subtarget(&TM.getSubtarget<ARMSubtarget>()) {
}

ARMSelectionDAGInfo::~ARMSelectionDAGInfo() {
}

SDValue
ARMSelectionDAGInfo::EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                             SDValue Chain,
                                             SDValue Dst, SDValue Src,
                                             SDValue Size, unsigned Align,
                                             bool isVolatile, bool AlwaysInline,
                                             MachinePointerInfo DstPtrInfo,
                                          MachinePointerInfo SrcPtrInfo) const {
  // Do repeated 4-byte loads and stores. To be improved.
  // This requires 4-byte alignment.
  if ((Align & 3) != 0)
    return SDValue();
  // This requires the copy size to be a constant, preferably
  // within a subtarget-specific limit.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();
  uint64_t SizeVal = ConstantSize->getZExtValue();
  if (!AlwaysInline && SizeVal > Subtarget->getMaxInlineSizeThreshold())
    return SDValue();

  unsigned BytesLeft = SizeVal & 3;
  unsigned NumMemOps = SizeVal >> 2;
  unsigned EmittedNumMemOps = 0;
  EVT VT = MVT::i32;
  unsigned VTSize = 4;
  unsigned i = 0;
  const unsigned MAX_LOADS_IN_LDM = 6;
  SDValue TFOps[MAX_LOADS_IN_LDM];
  SDValue Loads[MAX_LOADS_IN_LDM];
  uint64_t SrcOff = 0, DstOff = 0;

  // Emit up to MAX_LOADS_IN_LDM loads, then a TokenFactor barrier, then the
  // same number of stores.  The loads and stores will get combined into
  // ldm/stm later on.
  while (EmittedNumMemOps < NumMemOps) {
    for (i = 0;
         i < MAX_LOADS_IN_LDM && EmittedNumMemOps + i < NumMemOps; ++i) {
      Loads[i] = DAG.getLoad(VT, dl, Chain,
                             DAG.getNode(ISD::ADD, dl, MVT::i32, Src,
                                         DAG.getConstant(SrcOff, MVT::i32)),
                             SrcPtrInfo.getWithOffset(SrcOff), isVolatile,
                             false, false, 0);
      TFOps[i] = Loads[i].getValue(1);
      SrcOff += VTSize;
    }
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

    for (i = 0;
         i < MAX_LOADS_IN_LDM && EmittedNumMemOps + i < NumMemOps; ++i) {
      TFOps[i] = DAG.getStore(Chain, dl, Loads[i],
                              DAG.getNode(ISD::ADD, dl, MVT::i32, Dst,
                                          DAG.getConstant(DstOff, MVT::i32)),
                              DstPtrInfo.getWithOffset(DstOff),
                              isVolatile, false, 0);
      DstOff += VTSize;
    }
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

    EmittedNumMemOps += i;
  }

  if (BytesLeft == 0)
    return Chain;

  // Issue loads / stores for the trailing (1 - 3) bytes.
  unsigned BytesLeftSave = BytesLeft;
  i = 0;
  while (BytesLeft) {
    if (BytesLeft >= 2) {
      VT = MVT::i16;
      VTSize = 2;
    } else {
      VT = MVT::i8;
      VTSize = 1;
    }

    Loads[i] = DAG.getLoad(VT, dl, Chain,
                           DAG.getNode(ISD::ADD, dl, MVT::i32, Src,
                                       DAG.getConstant(SrcOff, MVT::i32)),
                           SrcPtrInfo.getWithOffset(SrcOff),
                           false, false, false, 0);
    TFOps[i] = Loads[i].getValue(1);
    ++i;
    SrcOff += VTSize;
    BytesLeft -= VTSize;
  }
  Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);

  i = 0;
  BytesLeft = BytesLeftSave;
  while (BytesLeft) {
    if (BytesLeft >= 2) {
      VT = MVT::i16;
      VTSize = 2;
    } else {
      VT = MVT::i8;
      VTSize = 1;
    }

    TFOps[i] = DAG.getStore(Chain, dl, Loads[i],
                            DAG.getNode(ISD::ADD, dl, MVT::i32, Dst,
                                        DAG.getConstant(DstOff, MVT::i32)),
                            DstPtrInfo.getWithOffset(DstOff), false, false, 0);
    ++i;
    DstOff += VTSize;
    BytesLeft -= VTSize;
  }
  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &TFOps[0], i);
}

// Adjust parameters for memset, EABI uses format (ptr, size, value),
// GNU library uses (ptr, value, size)
// See RTABI section 4.3.4
SDValue ARMSelectionDAGInfo::
EmitTargetCodeForMemset(SelectionDAG &DAG, DebugLoc dl,
                        SDValue Chain, SDValue Dst,
                        SDValue Src, SDValue Size,
                        unsigned Align, bool isVolatile,
                        MachinePointerInfo DstPtrInfo) const {
  // Use default for non AAPCS (or Darwin) subtargets
  if (!Subtarget->isAAPCS_ABI() || Subtarget->isTargetDarwin())
    return SDValue();

  const ARMTargetLowering &TLI =
    *static_cast<const ARMTargetLowering*>(DAG.getTarget().getTargetLowering());
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;

  // First argument: data pointer
  Type *IntPtrTy = TLI.getDataLayout()->getIntPtrType(*DAG.getContext());
  Entry.Node = Dst;
  Entry.Ty = IntPtrTy;
  Args.push_back(Entry);

  // Second argument: buffer size
  Entry.Node = Size;
  Entry.Ty = IntPtrTy;
  Entry.isSExt = false;
  Args.push_back(Entry);

  // Extend or truncate the argument to be an i32 value for the call.
  if (Src.getValueType().bitsGT(MVT::i32))
    Src = DAG.getNode(ISD::TRUNCATE, dl, MVT::i32, Src);
  else
    Src = DAG.getNode(ISD::ZERO_EXTEND, dl, MVT::i32, Src);

  // Third argument: value to fill
  Entry.Node = Src;
  Entry.Ty = Type::getInt32Ty(*DAG.getContext());
  Entry.isSExt = true;
  Args.push_back(Entry);

  // Emit __eabi_memset call
  TargetLowering::CallLoweringInfo CLI(Chain,
                    Type::getVoidTy(*DAG.getContext()), // return type
                    false, // return sign ext
                    false, // return zero ext
                    false, // is var arg
                    false, // is in regs
                    0,     // number of fixed arguments
                    TLI.getLibcallCallingConv(RTLIB::MEMSET), // call conv
                    false, // is tail call
                    false, // does not return
                    false, // is return val used
                    DAG.getExternalSymbol(TLI.getLibcallName(RTLIB::MEMSET),
                                          TLI.getPointerTy()), // callee
                    Args, DAG, dl);
  std::pair<SDValue,SDValue> CallResult =
    TLI.LowerCallTo(CLI);

  return CallResult.second;
}
