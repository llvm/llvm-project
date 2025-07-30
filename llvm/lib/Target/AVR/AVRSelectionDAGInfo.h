//===-- AVRSelectionDAGInfo.h - AVR SelectionDAG Info -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AVR subclass for SelectionDAGTargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AVR_SELECTION_DAG_INFO_H
#define LLVM_AVR_SELECTION_DAG_INFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "AVRGenSDNodeInfo.inc"

namespace llvm {

/// Holds information about the AVR instruction selection DAG.
class AVRSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  AVRSelectionDAGInfo();

  ~AVRSelectionDAGInfo() override;
};

} // end namespace llvm

#endif // LLVM_AVR_SELECTION_DAG_INFO_H
