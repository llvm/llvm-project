//===-- X86SelectionDAGInfo.h - X86 SelectionDAG Info -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef X86SELECTIONDAGINFO_H
#define X86SELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class X86TargetLowering;
class X86TargetMachine;
class X86Subtarget;

class X86SelectionDAGInfo : public TargetSelectionDAGInfo {
  /// Subtarget - Keep a pointer to the X86Subtarget around so that we can
  /// make the right decision when generating code for different targets.
  const X86Subtarget *Subtarget;

  const X86TargetLowering &TLI;

public:
  explicit X86SelectionDAGInfo(const X86TargetMachine &TM);
  ~X86SelectionDAGInfo();

  virtual
  SDValue EmitTargetCodeForMemset(SelectionDAG &DAG, DebugLoc dl,
                                  SDValue Chain,
                                  SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align,
                                  bool isVolatile,
                                  MachinePointerInfo DstPtrInfo) const;

  virtual
  SDValue EmitTargetCodeForMemcpy(SelectionDAG &DAG, DebugLoc dl,
                                  SDValue Chain,
                                  SDValue Dst, SDValue Src,
                                  SDValue Size, unsigned Align,
                                  bool isVolatile, bool AlwaysInline,
                                  MachinePointerInfo DstPtrInfo,
                                  MachinePointerInfo SrcPtrInfo) const;
};

}

#endif
