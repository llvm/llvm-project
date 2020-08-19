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

SDValue DPUSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, unsigned Align, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  if (AlwaysInline) {
    return SDValue();
  }

  bool DstIsMramPtr = DstPtrInfo.getAddrSpace() == DPUADDR_SPACE::MRAM;
  bool SrcIsMramPtr = SrcPtrInfo.getAddrSpace() == DPUADDR_SPACE::MRAM;

  if (DstIsMramPtr) {
    if (SrcIsMramPtr) {
      return EmitMemFnCall("__memcpy_mm", DAG, dl, Chain, Dst, Src, Size,
                           RTLIB::MEMCPY);
    }

    return EmitMemFnCall("__memcpy_mw", DAG, dl, Chain, Dst, Src, Size,
                         RTLIB::MEMCPY);
  } else if (SrcIsMramPtr) {
    return EmitMemFnCall("__memcpy_wm", DAG, dl, Chain, Dst, Src, Size,
                         RTLIB::MEMCPY);
  }

  return SDValue();
}

SDValue DPUSelectionDAGInfo::EmitTargetCodeForMemmove(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, unsigned Align, bool isVolatile,
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
    SDValue Size, unsigned Align, bool isVolatile,
    MachinePointerInfo DstPtrInfo) const {
  bool DstIsMramPtr = DstPtrInfo.getAddrSpace() == DPUADDR_SPACE::MRAM;

  if (DstIsMramPtr) {
    return EmitMemFnCall("__memset_m", DAG, dl, Chain, Dst, Src, Size,
                         RTLIB::MEMSET);
  }

  return SDValue();
}

} // namespace llvm
