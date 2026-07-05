//===- DeadCodeElimination.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Eliminate dead iterations.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_DEADCODEELIMINATION_H
#define POLLY_DEADCODEELIMINATION_H

#include "polly/DependenceInfo.h"

namespace polly {

bool runDeadCodeElim(Scop &S, DependenceAnalysis::Result &DA);
} // namespace polly

#endif /* POLLY_DEADCODEELIMINATION_H */
