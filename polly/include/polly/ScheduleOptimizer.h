//===- polly/ScheduleOptimizer.h - The Schedule Optimizer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCHEDULEOPTIMIZER_H
#define POLLY_SCHEDULEOPTIMIZER_H

#include "polly/DependenceInfo.h"

namespace llvm {
class TargetTransformInfo;
}

namespace polly {

void runIslScheduleOptimizer(Scop &S, llvm::TargetTransformInfo *TTI,
                             DependenceAnalysis::Result &Deps);
} // namespace polly

#endif // POLLY_SCHEDULEOPTIMIZER_H
