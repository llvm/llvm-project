//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_PPCSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_POWERPC_PPCSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

namespace llvm {

class PPCSelectionDAGInfo : public SelectionDAGTargetInfo {
public:
  ~PPCSelectionDAGInfo() override;

  bool isTargetMemoryOpcode(unsigned Opcode) const override;

  bool isTargetStrictFPOpcode(unsigned Opcode) const override;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_POWERPC_PPCSELECTIONDAGINFO_H
