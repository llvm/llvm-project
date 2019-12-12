//===---- hip_atomics.h - Declarations of hip atomic functions ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_AMDGCN_HIP_ATOMICS_H
#define OMPTARGET_AMDGCN_HIP_ATOMICS_H

#include "target_impl.h"

DEVICE unsigned atomicAdd(unsigned *address, unsigned val);
DEVICE int atomicAdd(int *address, int val);
DEVICE unsigned long long atomicAdd(unsigned long long *address,
                                    unsigned long long val);

DEVICE unsigned atomicInc(unsigned *address);
DEVICE unsigned atomicInc(unsigned *address, unsigned max);
DEVICE int atomicInc(int *address);

DEVICE int atomicMax(int *address, int val);
DEVICE unsigned atomicMax(unsigned *address, unsigned val);
DEVICE unsigned long long atomicMax(unsigned long long *address,
                                    unsigned long long val);

DEVICE int atomicExch(int *address, int val);
DEVICE unsigned atomicExch(unsigned *address, unsigned val);
DEVICE unsigned long long atomicExch(unsigned long long *address,
                                     unsigned long long val);

DEVICE unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val);
DEVICE int atomicCAS(int *address, int compare, int val);
DEVICE unsigned long long atomicCAS(unsigned long long *address,
                                    unsigned long long compare,
                                    unsigned long long val);

#endif
