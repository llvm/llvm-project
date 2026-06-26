//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LANAI_LANAIISELDAGTODAG_H
#define LLVM_LIB_TARGET_LANAI_LANAIISELDAGTODAG_H

#include "llvm/CodeGen/SelectionDAGISel.h"

namespace llvm {

class LanaiTargetMachine;

class LanaiISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  LanaiISelDAGToDAGPass(LanaiTargetMachine &TM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_LANAI_LANAIISELDAGTODAG_H
