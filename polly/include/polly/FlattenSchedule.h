//===------ FlattenSchedule.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Try to reduce the number of scatter dimension. Useful to make isl_union_map
// schedules more understandable. This is only intended for debugging and
// unittests, not for optimizations themselves.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_FLATTENSCHEDULE_H
#define POLLY_FLATTENSCHEDULE_H

namespace llvm {
class PassRegistry;
class Pass;
class raw_ostream;
} // namespace llvm

namespace polly {
llvm::Pass *createFlattenSchedulePass();
llvm::Pass *createFlattenSchedulePrinterLegacyPass(llvm::raw_ostream &OS);
} // namespace polly

namespace llvm {
void initializeFlattenSchedulePass(llvm::PassRegistry &);
void initializeFlattenSchedulePrinterLegacyPassPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_FLATTENSCHEDULE_H */
