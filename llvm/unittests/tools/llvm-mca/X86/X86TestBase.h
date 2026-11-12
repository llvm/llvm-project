//===---- X86TestBase.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test fixture common to all X86 MCA tests.
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TOOLS_LLVMMCA_X86_X86TESTBASE_H
#define LLVM_UNITTESTS_TOOLS_LLVMMCA_X86_X86TESTBASE_H

#include "MCATestBase.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace mca {

class X86TestBase : public MCATestBase {
protected:
  X86TestBase();

  void getSimpleInsts(SmallVectorImpl<MCInst> &Insts, unsigned Repeats = 1);
};

} // end namespace mca
} // end namespace llvm

#endif
