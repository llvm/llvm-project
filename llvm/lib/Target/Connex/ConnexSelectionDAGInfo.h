//===-- ConnexSelectionDAGInfo.h - Connex SelectionDAG Info -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the Connex subclass for SelectionDAGTargetInfo.
///
//===----------------------------------------------------------------------===//

// Inspired from ARM/ARMSelectionDAGInfo.cpp

#ifndef LLVM_LIB_TARGET_CONNEX_CONNEXSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_CONNEX_CONNEXSELECTIONDAGINFO_H

// Inspired from ARM/ARMSelectionDAGInfo.cpp (Oct 2025)
#include "llvm/CodeGen/RuntimeLibcallUtil.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

namespace llvm {

class ConnexSelectionDAGInfo : public SelectionDAGTargetInfo {
public:
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

  // Adjust parameters for memset, see RTABI section 4.3.4
  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Op1, SDValue Op2,
                                  SDValue Op3, Align Alignment, bool isVolatile,
                                  bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo) const override;

  SDValue EmitSpecializedLibcall(SelectionDAG &DAG, const SDLoc &dl,
                                 SDValue Chain, SDValue Dst, SDValue Src,
                                 SDValue Size, unsigned Align,
                                 RTLIB::Libcall LC) const;
};

} // end namespace llvm

#endif
