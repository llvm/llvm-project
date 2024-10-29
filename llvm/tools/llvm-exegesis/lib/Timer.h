//===---------- Timer.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_TIMER_H
#define LLVM_TOOLS_LLVM_EXEGESIS_TIMER_H

namespace llvm {
namespace exegesis {
extern bool TimerIsEnabled;

extern const char TimerGroupName[];
extern const char TimerGroupDescription[];

} // namespace exegesis
} // namespace llvm
#endif
