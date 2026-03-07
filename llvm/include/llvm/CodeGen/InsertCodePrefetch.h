//===- BasicBlockSectionUtils.h - Utilities for basic block sections     --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INSERTCODEPREFETCH_H
#define LLVM_CODEGEN_INSERTCODEPREFETCH_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/UniqueBBID.h"

namespace llvm {

SmallString<128> getPrefetchTargetSymbolName(StringRef FunctionName,
                                             const UniqueBBID &BBID,
                                             unsigned SubblockIndex);

} // end namespace llvm

#endif // LLVM_CODEGEN_INSERTCODEPREFETCH_H
