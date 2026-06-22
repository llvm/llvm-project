//===- ConstantExprLower.h - Lower ConstantExprs to instructions *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_CONSTANTEXPRLOWER_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_CONSTANTEXPRLOWER_H

#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

void lowerConstantExprs(llvm::Module &M);

}
}

#endif
