//===-- DPUSelectionDAGInfo.h - DPU SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DPU subclass for SelectionDAGTargetInfo.
//
// Purpose of the overloaded functions is to replace regular calls to
// intrinsic standard library functions to the underlying assembly code
// that directly invokes the functions. Otherwise, the linker will fail
// hooking the functions properly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_DPU_DPUSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

using namespace llvm;

class DPUSelectionDAGInfo : public SelectionDAGTargetInfo {
public:
  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align, bool isVolatile,
                                  bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const override;

  SDValue
  EmitTargetCodeForMemmove(SelectionDAG &DAG, const SDLoc &dl, SDValue Chain,
                           SDValue Dst, SDValue Src, SDValue Size,
                           unsigned Align, bool isVolatile,
                           MachinePointerInfo DstPtrInfo,
                           MachinePointerInfo SrcPtrInfo) const override;

  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align, bool isVolatile,
                                  MachinePointerInfo DstPtrInfo) const override;
};

#endif
