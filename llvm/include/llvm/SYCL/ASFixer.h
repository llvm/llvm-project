//===- ASFixer.h - SYCL address spaces fixer pass -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SYCL address spaces fixer pass
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_ASFIXER_H
#define LLVM_SYCL_ASFIXER_H

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {

ModulePass *createASFixerPass();

}

#endif
