//===-- DPUSelectionDAGInfo.cpp - DPU SelectionDAG Info -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DPUSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "DPUSelectionDAGInfo.h"

#include "DPUISelLowering.h"
#include "DPUTargetMachine.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"

#define DEBUG_TYPE "dpu-selectiondag-info"

namespace llvm {

static const uint32_t MramAlignment = 8;
static const uint32_t WramAlignment = 4;

static SDValue EmitMemFnCall(const char *FunctionName, SelectionDAG &DAG,
                             const SDLoc &dl, SDValue Chain, SDValue Dst,
                             SDValue Src, SDValue Size, RTLIB::Libcall LC) {
  const TargetLowering *TLI = DAG.getSubtarget().getTargetLowering();
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = Dst;
  Entry.Ty = DAG.getDataLayout().getIntPtrType(*DAG.getContext());
  Args.push_back(Entry);
  Entry.Node = Src;
  Args.push_back(Entry);
  Entry.Node = Size;
  Args.push_back(Entry);

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setLibCallee(TLI->getLibcallCallingConv(LC),
                    Type::getVoidTy(*DAG.getContext()),
                    DAG.getExternalSymbol(
                        FunctionName, TLI->getPointerTy(DAG.getDataLayout())),
                    std::move(Args))
      .setDiscardResult();

  std::pair<SDValue, SDValue> CallResult = TLI->LowerCallTo(CLI);
  return CallResult.second;
}

static SDValue EmitMemcpyMW(SelectionDAG &DAG, const SDLoc &dl, SDValue Chain,
                            SDValue Mram, SDValue Wram, int32_t DstAlign,
                            int32_t SrcAlign, SDValue Size,
                            bool CanFetchConstant, uint64_t Length) {
  if ((SrcAlign == MramAlignment) && (DstAlign == MramAlignment) &&
      CanFetchConstant && properDMASize(Length)) {
    const DPUTargetLowering &DTL =
        static_cast<const DPUTargetLowering &>(DAG.getTargetLoweringInfo());
    return DTL.LowerDMAUnchecked(DAG, dl, Mram.getValueType(), Chain, Wram,
                                 Mram, Size, CanFetchConstant, Length,
                                 DPUISD::SDMA);
  } else {
    return EmitMemFnCall("__memcpy_mw", DAG, dl, Chain, Mram, Wram, Size,
                         RTLIB::MEMCPY);
  }
}

static SDValue EmitMemcpyWM(SelectionDAG &DAG, const SDLoc &dl, SDValue Chain,
                            SDValue Wram, SDValue Mram, int32_t DstAlign,
                            int32_t SrcAlign, SDValue Size,
                            bool CanFetchConstant, uint64_t Length) {
  if ((SrcAlign == MramAlignment) && (DstAlign == MramAlignment) &&
      CanFetchConstant && properDMASize(Length)) {
    const DPUTargetLowering &DTL =
        static_cast<const DPUTargetLowering &>(DAG.getTargetLoweringInfo());
    return DTL.LowerDMAUnchecked(DAG, dl, Wram.getValueType(), Chain, Wram,
                                 Mram, Size, CanFetchConstant, Length,
                                 DPUISD::LDMA);
  } else {
    return EmitMemFnCall("__memcpy_wm", DAG, dl, Chain, Wram, Mram, Size,
                         RTLIB::MEMCPY);
  }
}

SDValue DPUSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  if (AlwaysInline) {
    return SDValue();
  }

  bool DstIsMramPtr = DstPtrInfo.getAddrSpace() == DPUADDR_SPACE::MRAM;
  bool SrcIsMramPtr = SrcPtrInfo.getAddrSpace() == DPUADDR_SPACE::MRAM;

  uint64_t Length;
  bool CanFetchConstant = canFetchConstantTo(Size, &Length);

  int32_t SrcAlign = getSDValueAlignment(Src);
  int32_t DstAlign = getSDValueAlignment(Dst);
  if ((int32_t)Alignment.value() > DstAlign)
    DstAlign = Alignment.value();

  if (DstIsMramPtr) {
    if (SrcIsMramPtr) {
      return EmitMemFnCall("__memcpy_mm", DAG, dl, Chain, Dst, Src, Size,
                           RTLIB::MEMCPY);
    }

    return EmitMemcpyMW(DAG, dl, Chain, Dst, Src, DstAlign, SrcAlign, Size,
                        CanFetchConstant, Length);
  } else if (SrcIsMramPtr) {
    return EmitMemcpyWM(DAG, dl, Chain, Dst, Src, DstAlign, SrcAlign, Size,
                        CanFetchConstant, Length);
  } else {
    if ((DstAlign % WramAlignment == 0) && (SrcAlign % WramAlignment == 0) &&
        CanFetchConstant && (Length % WramAlignment == 0)) {
      return EmitMemFnCall("__memcpy_wram_4align", DAG, dl, Chain, Dst, Src,
                           Size, RTLIB::MEMCPY);
    }
  }

  return SDValue();
}

SDValue DPUSelectionDAGInfo::EmitTargetCodeForMemmove(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  bool DstIsMramPtr = DstPtrInfo.getAddrSpace() == DPUADDR_SPACE::MRAM;
  bool SrcIsMramPtr = SrcPtrInfo.getAddrSpace() == DPUADDR_SPACE::MRAM;

  if (DstIsMramPtr) {
    if (SrcIsMramPtr) {
      return EmitMemFnCall("__memmove_mm", DAG, dl, Chain, Dst, Src, Size,
                           RTLIB::MEMMOVE);
    }

    return EmitMemFnCall("__memcpy_mw", DAG, dl, Chain, Dst, Src, Size,
                         RTLIB::MEMCPY);
  } else if (SrcIsMramPtr) {
    return EmitMemFnCall("__memcpy_wm", DAG, dl, Chain, Dst, Src, Size,
                         RTLIB::MEMCPY);
  }

  return SDValue();
}

SDValue DPUSelectionDAGInfo::EmitTargetCodeForMemset(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile,
    MachinePointerInfo DstPtrInfo) const {
  bool DstIsMramPtr = DstPtrInfo.getAddrSpace() == DPUADDR_SPACE::MRAM;

  uint64_t Length;
  bool CanFetchConstant = canFetchConstantTo(Size, &Length);

  int32_t DstAlign = getSDValueAlignment(Dst);
  if ((int32_t)Alignment.value() > DstAlign)
    DstAlign = Alignment.value();

  if (DstIsMramPtr) {
    if ((DstAlign % MramAlignment == 0) && CanFetchConstant &&
        (Length % MramAlignment == 0)) {
      return EmitMemFnCall("__memset_mram_8align", DAG, dl, Chain, Dst, Src,
                           Size, RTLIB::MEMSET);
    } else {
      return EmitMemFnCall("__memset_mram", DAG, dl, Chain, Dst, Src, Size,
                           RTLIB::MEMSET);
    }
  } else {
    if ((DstAlign % WramAlignment == 0) && CanFetchConstant &&
        (Length % WramAlignment == 0)) {
      return EmitMemFnCall("__memset_wram_4align", DAG, dl, Chain, Dst, Src,
                           Size, RTLIB::MEMSET);
    }
  }
  return SDValue();
}

} // namespace llvm
