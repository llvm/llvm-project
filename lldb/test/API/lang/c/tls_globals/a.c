//===-- a.c -----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unistd.h>

__thread int var_shared = 33;

int
touch_shared()
{
    return var_shared;
}

void shared_check()
{
	var_shared *= 2;
	usleep(1); // shared thread breakpoint
}
