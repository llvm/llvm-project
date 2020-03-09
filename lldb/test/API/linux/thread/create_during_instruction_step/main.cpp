//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file deliberately uses low level linux-specific API for thread creation because:
// - instruction-stepping over thread creation using higher-level functions was very slow
// - it was also unreliable due to single-stepping bugs unrelated to this test
// - some threading libraries do not create or destroy threads when we would expect them to

#include <sched.h>

#include <atomic>
#include <cstdio>

enum { STACK_SIZE = 0x2000 };

static uint8_t child_stack[STACK_SIZE];

pid_t child_tid;

std::atomic<bool> flag(false);

int thread_main(void *)
{
    while (! flag) // Make sure the thread does not exit prematurely
        ;

    return 0;
}

int main ()
{
    int ret = clone(thread_main,
            child_stack + STACK_SIZE/2, // Don't care whether the stack grows up or down,
                                        // just point to the middle
            CLONE_CHILD_CLEARTID | CLONE_FILES | CLONE_FS | CLONE_PARENT_SETTID |
            CLONE_SIGHAND | CLONE_SYSVSEM | CLONE_THREAD | CLONE_VM,
            nullptr, // thread_main argument
            &child_tid);

    if (ret == -1)
    {
        perror("clone");
        return 1;
    }

    flag = true;

    return 0;
}
