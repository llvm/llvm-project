//===-- HexagonSelectionDAGInfo.h - Hexagon SelectionDAG Info ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Hexagon subclass for TargetSelectionDAGInfo.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonSELECTIONDAGINFO_H
#define HexagonSELECTIONDAGINFO_H

#include "llvm/Target/TargetSelectionDAGInfo.h"

namespace llvm {

class HexagonTargetMachine;

class HexagonSelectionDAGInfo : public TargetSelectionDAGInfo {
public:
  explicit HexagonSelectionDAGInfo(const HexagonTargetMachine &TM);
  ~HexagonSelectionDAGInfo();

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
