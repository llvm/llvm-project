//===- FrozenRewritePatternList.cpp - Frozen Pattern List -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternList.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// FrozenRewritePatternList
//===----------------------------------------------------------------------===//

FrozenRewritePatternList::FrozenRewritePatternList(
    OwningRewritePatternList &&patterns)
    : patterns(patterns.takePatterns()) {}
