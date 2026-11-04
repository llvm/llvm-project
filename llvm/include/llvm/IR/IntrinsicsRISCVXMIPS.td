//===- IntrinsicsRISCVXMIPS.td - MIPS intrinsics -------*- tablegen -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the MIPS specific intrinsics for RISC-V.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {
  def int_riscv_mips_pause : ClangBuiltin<"__builtin_riscv_mips_pause">,
        Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
  def int_riscv_mips_ehb : ClangBuiltin<"__builtin_riscv_mips_ehb">,
        Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
  def int_riscv_mips_ihb : ClangBuiltin<"__builtin_riscv_mips_ihb">,
        Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
}
