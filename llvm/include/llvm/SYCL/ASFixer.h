//===- ASFixer.h - SYCL address spaces fixer pass -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
