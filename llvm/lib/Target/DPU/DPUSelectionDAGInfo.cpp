//===-- DPUSelectionDAGInfo.h - DPU SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DPUSelectionDAGInfo.h"
#include "DPUSubtarget.h"
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
  return SDValue();
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
  return SDValue();
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
  return SDValue();
}
