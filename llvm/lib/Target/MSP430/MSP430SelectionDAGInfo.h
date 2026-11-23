//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MSP430_MSP430SELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_MSP430_MSP430SELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "MSP430GenSDNodeInfo.inc"

namespace llvm {

class MSP430SelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  MSP430SelectionDAGInfo();

  ~MSP430SelectionDAGInfo() override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_MSP430_MSP430SELECTIONDAGINFO_H
