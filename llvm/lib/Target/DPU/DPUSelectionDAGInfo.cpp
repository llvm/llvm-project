//===-- DPUSelectionDAGInfo.h - DPU SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DPUSelectionDAGInfo.h"
#include "DPUISelLowering.h"
#include "DPUSubtarget.h"
#include "DPUTargetLowering.h"
#include <llvm/CodeGen/TargetLowering.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "dpu-lower"

SDValue DPUSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, unsigned Align, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  LLVM_DEBUG({
    dbgs() << "DPU/Lower - EmitTargetCodeForMemcpy";
    dbgs() << " dest = ";
    Dst->dump(&DAG);
    dbgs() << " src  = ";
    Src->dump(&DAG);
    dbgs() << " size = ";
    Size->dump(&DAG);
    dbgs() << " chain = ";
    Chain->dump(&DAG);
  });
  return EmitTargetCodeForIntrinsicCall("__intrinsic__memcpy", DAG, dl, Chain,
                                        Dst, Src, Size);
}

SDValue DPUSelectionDAGInfo::EmitTargetCodeForMemmove(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, unsigned Align, bool isVolatile,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  LLVM_DEBUG({
    dbgs() << "DPU/Lower - EmitTargetCodeForMemmove";
    dbgs() << " dest = ";
    Dst->dump(&DAG);
    dbgs() << " src  = ";
    Src->dump(&DAG);
    dbgs() << " size = ";
    Size->dump(&DAG);
    dbgs() << " chain = ";
    Chain->dump(&DAG);
  });
  return EmitTargetCodeForIntrinsicCall("__intrinsic__memmove", DAG, dl, Chain,
                                        Dst, Src, Size);
}

SDValue DPUSelectionDAGInfo::EmitTargetCodeForMemset(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, unsigned Align, bool isVolatile,
    MachinePointerInfo DstPtrInfo) const {
  LLVM_DEBUG({
    dbgs() << "DPU/Lower - EmitTargetCodeForMemset";
    dbgs() << " op1 = ";
    Dst->dump(&DAG);
    dbgs() << " op2  = ";
    Src->dump(&DAG);
    dbgs() << " op3 = ";
    Size->dump(&DAG);
    dbgs() << " chain = ";
    Chain->dump(&DAG);
  });
  return EmitTargetCodeForIntrinsicCall("__intrinsic__memset", DAG, dl, Chain,
                                        Dst, Src, Size);
}

SDValue DPUSelectionDAGInfo::EmitTargetCodeForIntrinsicCall(
    const char *IntrinsicFunctionName, SelectionDAG &DAG, SDLoc dl,
    SDValue Chain, SDValue Op1, SDValue Op2, SDValue Op3) const {
  // The principle is to rely as much as possible on the target lowering
  // process, since it handles a number of corner cases regarding arguments that
  // cannot be reasonably done there. The simplest way to do that is to rebuild
  // a function call and invoke LowerCallTo, itself invoking LowerCall in the
  // DPU target lowering. In order for LowerCall to identify that this is not a
  // "regular call" (generating a DPUISD::CALL) but an intrinsic one (generating
  // a DPUISD::INTRINSIC_CALL), we just use a very special function name, which,
  // in theory will never be used by the developers.
  const auto &Subtarget = DAG.getMachineFunction().getSubtarget<DPUSubtarget>();
  const DPUTargetLowering *TLI = Subtarget.getTargetLowering();

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Ty = DAG.getDataLayout().getIntPtrType(*DAG.getContext());
  Entry.Node = Op1;
  Args.push_back(Entry);

  Entry.Node = Op2;
  Args.push_back(Entry);

  Entry.Node = Op3;
  Args.push_back(Entry);

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl);
  CLI.setChain(Chain);
  CLI.setDiscardResult();
  CLI.setCallee(CallingConv::C, Type::getVoidTy(*DAG.getContext()),
                DAG.getExternalSymbol(IntrinsicFunctionName,
                                      TLI->getPointerTy(DAG.getDataLayout())),
                std::move(Args));
  return TLI->LowerCallTo(CLI).second;
}
