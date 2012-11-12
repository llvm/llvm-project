//===-- X86SelectionDAGInfo.cpp - X86 SelectionDAG Info -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86SelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "x86-selectiondag-info"
#include "X86TargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;

X86SelectionDAGInfo::X86SelectionDAGInfo(const X86TargetMachine &TM) :
  TargetSelectionDAGInfo(TM),
  Subtarget(&TM.getSubtarget<X86Subtarget>()),
  TLI(*TM.getTargetLowering()) {
}

X86SelectionDAGInfo::~X86SelectionDAGInfo() {
}

SDValue
X86SelectionDAGInfo::EmitTargetCodeForMemset(SelectionDAG &DAG, DebugLoc dl,
                                             SDValue Chain,
                                             SDValue Dst, SDValue Src,
                                             SDValue Size, unsigned Align,
                                             bool isVolatile,
                                         MachinePointerInfo DstPtrInfo) const {
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);

  // If to a segment-relative address space, use the default lowering.
  if (DstPtrInfo.getAddrSpace() >= 256)
    return SDValue();

  // If not DWORD aligned or size is more than the threshold, call the library.
  // The libc version is likely to be faster for these cases. It can use the
  // address value and run time information about the CPU.
  if ((Align & 3) != 0 ||
      !ConstantSize ||
      ConstantSize->getZExtValue() >
        Subtarget->getMaxInlineSizeThreshold()) {
    SDValue InFlag(0, 0);

    // Check to see if there is a specialized entry-point for memory zeroing.
    ConstantSDNode *V = dyn_cast<ConstantSDNode>(Src);

    if (const char *bzeroEntry =  V &&
        V->isNullValue() ? Subtarget->getBZeroEntry() : 0) {
      EVT IntPtr = TLI.getPointerTy();
      Type *IntPtrTy = getDataLayout()->getIntPtrType(*DAG.getContext());
      TargetLowering::ArgListTy Args;
      TargetLowering::ArgListEntry Entry;
      Entry.Node = Dst;
      Entry.Ty = IntPtrTy;
      Args.push_back(Entry);
      Entry.Node = Size;
      Args.push_back(Entry);
      TargetLowering::
      CallLoweringInfo CLI(Chain, Type::getVoidTy(*DAG.getContext()),
                        false, false, false, false,
                        0, CallingConv::C, /*isTailCall=*/false,
                        /*doesNotRet=*/false, /*isReturnValueUsed=*/false,
                        DAG.getExternalSymbol(bzeroEntry, IntPtr), Args,
                        DAG, dl);
      std::pair<SDValue,SDValue> CallResult =
        TLI.LowerCallTo(CLI);
      return CallResult.second;
    }

    // Otherwise have the target-independent code call memset.
    return SDValue();
  }

  uint64_t SizeVal = ConstantSize->getZExtValue();
  SDValue InFlag(0, 0);
  EVT AVT;
  SDValue Count;
  ConstantSDNode *ValC = dyn_cast<ConstantSDNode>(Src);
  unsigned BytesLeft = 0;
  bool TwoRepStos = false;
  if (ValC) {
    unsigned ValReg;
    uint64_t Val = ValC->getZExtValue() & 255;

    // If the value is a constant, then we can potentially use larger sets.
    switch (Align & 3) {
    case 2:   // WORD aligned
      AVT = MVT::i16;
      ValReg = X86::AX;
      Val = (Val << 8) | Val;
      break;
    case 0:  // DWORD aligned
      AVT = MVT::i32;
      ValReg = X86::EAX;
      Val = (Val << 8)  | Val;
      Val = (Val << 16) | Val;
      if (Subtarget->is64Bit() && ((Align & 0x7) == 0)) {  // QWORD aligned
        AVT = MVT::i64;
        ValReg = X86::RAX;
        Val = (Val << 32) | Val;
      }
      break;
    default:  // Byte aligned
      AVT = MVT::i8;
      ValReg = X86::AL;
      Count = DAG.getIntPtrConstant(SizeVal);
      break;
    }

    if (AVT.bitsGT(MVT::i8)) {
      unsigned UBytes = AVT.getSizeInBits() / 8;
      Count = DAG.getIntPtrConstant(SizeVal / UBytes);
      BytesLeft = SizeVal % UBytes;
    }

    Chain  = DAG.getCopyToReg(Chain, dl, ValReg, DAG.getConstant(Val, AVT),
                              InFlag);
    InFlag = Chain.getValue(1);
  } else {
    AVT = MVT::i8;
    Count  = DAG.getIntPtrConstant(SizeVal);
    Chain  = DAG.getCopyToReg(Chain, dl, X86::AL, Src, InFlag);
    InFlag = Chain.getValue(1);
  }

  Chain  = DAG.getCopyToReg(Chain, dl, Subtarget->is64Bit() ? X86::RCX :
                                                              X86::ECX,
                            Count, InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, dl, Subtarget->is64Bit() ? X86::RDI :
                                                              X86::EDI,
                            Dst, InFlag);
  InFlag = Chain.getValue(1);

  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue Ops[] = { Chain, DAG.getValueType(AVT), InFlag };
  Chain = DAG.getNode(X86ISD::REP_STOS, dl, Tys, Ops, array_lengthof(Ops));

  if (TwoRepStos) {
    InFlag = Chain.getValue(1);
    Count  = Size;
    EVT CVT = Count.getValueType();
    SDValue Left = DAG.getNode(ISD::AND, dl, CVT, Count,
                               DAG.getConstant((AVT == MVT::i64) ? 7 : 3, CVT));
    Chain  = DAG.getCopyToReg(Chain, dl, (CVT == MVT::i64) ? X86::RCX :
                                                             X86::ECX,
                              Left, InFlag);
    InFlag = Chain.getValue(1);
    Tys = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue Ops[] = { Chain, DAG.getValueType(MVT::i8), InFlag };
    Chain = DAG.getNode(X86ISD::REP_STOS, dl, Tys, Ops, array_lengthof(Ops));
  } else if (BytesLeft) {
    // Handle the last 1 - 7 bytes.
    unsigned Offset = SizeVal - BytesLeft;
    EVT AddrVT = Dst.getValueType();
    EVT SizeVT = Size.getValueType();

    Chain = DAG.getMemset(Chain, dl,
                          DAG.getNode(ISD::ADD, dl, AddrVT, Dst,
                                      DAG.getConstant(Offset, AddrVT)),
                          Src,
                          DAG.getConstant(BytesLeft, SizeVT),
                          Align, isVolatile, DstPtrInfo.getWithOffset(Offset));
  }

  // TODO: Use a Tokenfactor, as in memcpy, instead of a single chain.
  return Chain;
}

SDValue
X86SelectionDAGInfo::EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                        SDValue Chain, SDValue Dst, SDValue Src,
                                        SDValue Size, unsigned Align,
                                        bool isVolatile, bool AlwaysInline,
                                         MachinePointerInfo DstPtrInfo,
                                         MachinePointerInfo SrcPtrInfo) const {
  // This requires the copy size to be a constant, preferably
  // within a subtarget-specific limit.
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (!ConstantSize)
    return SDValue();
  uint64_t SizeVal = ConstantSize->getZExtValue();
  if (!AlwaysInline && SizeVal > Subtarget->getMaxInlineSizeThreshold())
    return SDValue();

  /// If not DWORD aligned, it is more efficient to call the library.  However
  /// if calling the library is not allowed (AlwaysInline), then soldier on as
  /// the code generated here is better than the long load-store sequence we
  /// would otherwise get.
  if (!AlwaysInline && (Align & 3) != 0)
    return SDValue();

  // If to a segment-relative address space, use the default lowering.
  if (DstPtrInfo.getAddrSpace() >= 256 ||
      SrcPtrInfo.getAddrSpace() >= 256)
    return SDValue();

  MVT AVT;
  if (Align & 1)
    AVT = MVT::i8;
  else if (Align & 2)
    AVT = MVT::i16;
  else if (Align & 4)
    // DWORD aligned
    AVT = MVT::i32;
  else
    // QWORD aligned
    AVT = Subtarget->is64Bit() ? MVT::i64 : MVT::i32;

  unsigned UBytes = AVT.getSizeInBits() / 8;
  unsigned CountVal = SizeVal / UBytes;
  SDValue Count = DAG.getIntPtrConstant(CountVal);
  unsigned BytesLeft = SizeVal % UBytes;

  SDValue InFlag(0, 0);
  Chain  = DAG.getCopyToReg(Chain, dl, Subtarget->is64Bit() ? X86::RCX :
                                                              X86::ECX,
                            Count, InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, dl, Subtarget->is64Bit() ? X86::RDI :
                                                              X86::EDI,
                            Dst, InFlag);
  InFlag = Chain.getValue(1);
  Chain  = DAG.getCopyToReg(Chain, dl, Subtarget->is64Bit() ? X86::RSI :
                                                              X86::ESI,
                            Src, InFlag);
  InFlag = Chain.getValue(1);

  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue Ops[] = { Chain, DAG.getValueType(AVT), InFlag };
  SDValue RepMovs = DAG.getNode(X86ISD::REP_MOVS, dl, Tys, Ops,
                                array_lengthof(Ops));

  SmallVector<SDValue, 4> Results;
  Results.push_back(RepMovs);
  if (BytesLeft) {
    // Handle the last 1 - 7 bytes.
    unsigned Offset = SizeVal - BytesLeft;
    EVT DstVT = Dst.getValueType();
    EVT SrcVT = Src.getValueType();
    EVT SizeVT = Size.getValueType();
    Results.push_back(DAG.getMemcpy(Chain, dl,
                                    DAG.getNode(ISD::ADD, dl, DstVT, Dst,
                                                DAG.getConstant(Offset, DstVT)),
                                    DAG.getNode(ISD::ADD, dl, SrcVT, Src,
                                                DAG.getConstant(Offset, SrcVT)),
                                    DAG.getConstant(BytesLeft, SizeVT),
                                    Align, isVolatile, AlwaysInline,
                                    DstPtrInfo.getWithOffset(Offset),
                                    SrcPtrInfo.getWithOffset(Offset)));
  }

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                     &Results[0], Results.size());
}
