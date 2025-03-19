//===-- Kernel.td - Kernel definitions for Offload ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Offload API definitions related to the kernel handle
//
//===----------------------------------------------------------------------===//

def : Function {
    let name = "olCreateKernel";
    let desc = "Create a kernel from the function identified by `KernelName` in the given program.";
    let details = [];
    let params = [
        Param<"ol_program_handle_t", "Program", "handle of the program", PARAM_IN>,
        Param<"const char*", "KernelName", "name of the kernel entry point in the program", PARAM_IN>,
        Param<"ol_kernel_handle_t*", "Kernel", "output pointer for the created kernel", PARAM_OUT>
    ];
    let returns = [];
}

def : Function {
    let name = "olDestroyKernel";
    let desc = "Destroy the kernel and free all underlying resources.";
    let details = [];
    let params = [
        Param<"ol_kernel_handle_t", "Kernel", "handle of the kernel", PARAM_IN>
    ];
    let returns = [];
}
