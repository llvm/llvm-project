//===------- Profiling.cpp ---------------------------------------- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Profiling.h"

#pragma omp begin declare target device_type(nohost)

extern "C" {

void __llvm_profile_register_function(void *ptr) {}
void __llvm_profile_register_names_function(void *ptr, long int i) {}
}

#pragma omp end declare target
