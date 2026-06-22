//===- BitcodeEmitter.h - Metal v1 bitcode emitter --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_BITCODEEMITTER_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_BITCODEEMITTER_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"
#include <vector>

namespace llvm {
namespace metal {

std::vector<uint8_t> emitAIRBitcode(llvm::Module &M, PointeeTypeMap &PTM);

}
}

#endif
