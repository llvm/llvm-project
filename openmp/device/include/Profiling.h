//===-------- Profiling.h - OpenMP interface ---------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_DEVICERTL_PROFILING_H
#define OMPTARGET_DEVICERTL_PROFILING_H

extern "C" {
void __llvm_profile_register_function(void *Ptr);
void __llvm_profile_register_names_function(void *Ptr, long int I);
void __llvm_profile_instrument_memop(long int I, void *Ptr, int I2);
}

#endif
