//===-- Next32SelectionDAGInfo.cpp - Next32 SelectionDAG Info -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the Next32 subclass for SelectionDAGTargetInfo.
//
//===----------------------------------------------------------------------===//

#include "Next32TargetMachine.h"
#include "llvm/CodeGen/RuntimeLibcallUtil.h"

using namespace llvm;

#define DEBUG_TYPE "next32-selectiondag-info"

SDValue Next32SelectionDAGInfo::emitTargetCodeForLibcall(
    SelectionDAG &DAG, const SDLoc &dl, RTLIB::Libcall LC, SDValue Chain,
    SDValue Dst, Type *DstTy, SDValue Src, Type *SrcTy, SDValue Size) const {
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;

  Entry.Node = Dst;
  Entry.Ty = DstTy;
  Args.push_back(Entry);

  Entry.Node = Src;
  Entry.Ty = SrcTy;
  Args.push_back(Entry);

  // The default lowering path forces "Size" to be pointer-sized, assuming that
  // target's registers are at least as wide as its address space. This is not
  // the case with Next32, where registers are 32 bits wide, but the address
  // space is 64 bits wide, and thus, pointer-sized values are held in two
  // registers. In case of Next32, forcing "Size" argument which is smaller than
  // 64 bits to be pointer-sized makes the compiler try to split it into two
  // 32-bit registers, which is incorrect. To work around this, we force "Size"
  // to be register-sized when it's 32 bits or smaller.
  Entry.Ty = DAG.getDataLayout().getLargestLegalIntType(*DAG.getContext());
  Entry.Node = Size;
  Args.push_back(Entry);

  const TargetLowering &TLI = *DAG.getSubtarget().getTargetLowering();
  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setLibCallee(
          TLI.getLibcallCallingConv(LC),
          Dst.getValueType().getTypeForEVT(*DAG.getContext()),
          DAG.getExternalSymbol(TLI.getLibcallName(LC),
                                TLI.getPointerTy(DAG.getDataLayout())),
          std::move(Args))
      .setDiscardResult();

  std::pair<SDValue, SDValue> CallResult = TLI.LowerCallTo(CLI);
  return CallResult.second;
}

SDValue Next32SelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  // Only handle cases where "Size" argument is 32 bits or smaller.
  if (Size.getValueSizeInBits() > 32)
    return SDValue();

  Type *DstTy = PointerType::getUnqual(*DAG.getContext());
  Type *SrcTy = DstTy;
  return emitTargetCodeForLibcall(DAG, dl, RTLIB::MEMCPY, Chain, Dst, DstTy,
                                  Src, SrcTy, Size);
}

SDValue Next32SelectionDAGInfo::EmitTargetCodeForMemmove(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  // Only handle cases where "Size" argument is 32 bits or smaller.
  if (Size.getValueSizeInBits() > 32)
    return SDValue();

  Type *DstTy = PointerType::getUnqual(*DAG.getContext());
  Type *SrcTy = DstTy;
  return emitTargetCodeForLibcall(DAG, dl, RTLIB::MEMMOVE, Chain, Dst, DstTy,
                                  Src, SrcTy, Size);
}

SDValue Next32SelectionDAGInfo::EmitTargetCodeForMemset(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo) const {
  // Only handle cases where "Size" argument is 32 bits or smaller.
  if (Size.getValueSizeInBits() > 32)
    return SDValue();

  Type *DstTy = PointerType::getUnqual(*DAG.getContext());
  Type *SrcTy = Src.getValueType().getTypeForEVT(*DAG.getContext());
  return emitTargetCodeForLibcall(DAG, dl, RTLIB::MEMSET, Chain, Dst, DstTy,
                                  Src, SrcTy, Size);
}