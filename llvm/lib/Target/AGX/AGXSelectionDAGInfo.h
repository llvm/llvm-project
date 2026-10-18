//===-- AGXSelectionDAGInfo.h - AGX SelectionDAG info -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_AGX_AGXSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "AGXGenSDNodeInfo.inc"

namespace llvm {

class AGXSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  AGXSelectionDAGInfo();
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXSELECTIONDAGINFO_H
