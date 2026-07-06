//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSASELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_XTENSA_XTENSASELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "XtensaGenSDNodeInfo.inc"

namespace llvm {

class XtensaSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  XtensaSelectionDAGInfo();

  ~XtensaSelectionDAGInfo() override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_XTENSA_XTENSASELECTIONDAGINFO_H
