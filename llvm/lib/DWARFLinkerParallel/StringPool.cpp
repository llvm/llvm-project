//=== StringPool.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinkerParallel/StringPool.h"

thread_local llvm::BumpPtrAllocator
    llvm::dwarflinker_parallel::PerThreadStringAllocator::ThreadLocalAllocator;
