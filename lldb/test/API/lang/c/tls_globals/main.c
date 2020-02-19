//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void shared_check();
// On some OS's (darwin) you must actually access a thread local variable
// before you can read it
int
touch_shared();

// Create some TLS storage within the static executable.
__thread int var_static = 44;

void *fn_static(void *param)
{
	var_static *= 2;
	shared_check();
	usleep(1); // thread breakpoint
	for(;;)
		usleep(1);
}

int main (int argc, char const *argv[])
{
	pthread_t handle;
	pthread_create(&handle, NULL, &fn_static, NULL);
        touch_shared();
        for (; var_static;)
        {
            usleep(1); // main breakpoint
        }

        return 0;
}
