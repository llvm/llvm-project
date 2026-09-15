//===----- SplitModuleCommon.h - shared module splitting helpers ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Common utilities shared by the module splitting implementations
// (SplitModule, AMDGPUSplitModule, ...).
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SPLITMODULECOMMON_H
#define LLVM_TRANSFORMS_UTILS_SPLITMODULECOMMON_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class GlobalValue;

/// Assign a stable name to \p GV if it is unnamed, so that it is named
/// consistently across the split partitions. setName will give a distinct
/// name (e.g. __llvmsplit_externalize_unnamed.1) to each such entity.
LLVM_ABI void nameUnnamedGlobalValue(GlobalValue &GV);

/// If \p GV has local linkage, promote it to external + hidden visibility so
/// it can be referenced across module partitions, and assign a stable name
/// to unnamed entities (see nameUnnamedGlobalValue).
LLVM_ABI void externalizeGlobal(GlobalValue &GV);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SPLITMODULECOMMON_H
