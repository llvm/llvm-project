//===- AccCompilerAPI.td - OpenACC compiler-facing entry points -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Symbols the compiler emits calls to.
//
//===----------------------------------------------------------------------===//

def : AccExport<"__tgt_acc_rtl_init">;
def : AccExport<"__tgt_acc_rtl_deinit">;
