//===------ FlattenSchedule.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
} // anonymous namespace

namespace polly {
llvm::Pass *createFlattenSchedulePass();
} // namespace polly

namespace llvm {
void initializeFlattenSchedulePass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_FLATTENSCHEDULE_H */
