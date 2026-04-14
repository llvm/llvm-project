//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains common utilities for code prefetch insertion.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_INSERTCODEPREFETCH_H
#define LLVM_CODEGEN_INSERTCODEPREFETCH_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/UniqueBBID.h"

namespace llvm {

// Returns the symbol name for a prefetch target at function `FunctionName`,
// basic block `BBID` and callsite index `CallsiteIndex`.
SmallString<128> getPrefetchTargetSymbolName(StringRef FunctionName,
                                             const UniqueBBID &BBID,
                                             unsigned CallsiteIndex);

} // end namespace llvm

#endif // LLVM_CODEGEN_INSERTCODEPREFETCH_H
