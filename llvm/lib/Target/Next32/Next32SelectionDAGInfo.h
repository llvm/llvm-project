//===-- Next32SelectionDAGInfo.h - Next32 SelectionDAG Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Next32 subclass for SelectionDAGTargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_NEXT32SELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_NEXT32_NEXT32SELECTIONDAGINFO_H

#include "llvm/CodeGen/RuntimeLibcallUtil.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

namespace llvm {

class Next32SelectionDAGInfo : public SelectionDAGTargetInfo {
private:
  SDValue emitTargetCodeForLibcall(SelectionDAG &DAG, const SDLoc &dl,
                                   RTLIB::Libcall LC, SDValue Chain,
                                   SDValue Dst, Type *DstTy, SDValue Src,
                                   Type *SrcTy, SDValue Size) const;

public:
  /// Next32 is peculiar in that it's an architecture with a 64-bit address
  /// space, but 32-bit registers. This makes it impossible to correctly lower
  /// memcpy/memmove/memset calls with the "Size" argument smaller than 64 bits.

  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const override;

  SDValue
  EmitTargetCodeForMemmove(SelectionDAG &DAG, const SDLoc &dl, SDValue Chain,
                           SDValue Dst, SDValue Src, SDValue Size,
                           Align Alignment, bool isVolatile,
                           MachinePointerInfo DstPtrInfo,
                           MachinePointerInfo SrcPtrInfo) const override;

  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Dst, SDValue Src,
                                  SDValue Size, Align Alignment,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo) const override;
};

} // namespace llvm

#endif