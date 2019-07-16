// WebAssemblyInstrRef.td - WebAssembly reference type codegen --*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly refence type operand codegen constructs.
///
//===----------------------------------------------------------------------===//

defm SELECT_EXNREF : I<(outs EXNREF:$dst),
                       (ins EXNREF:$lhs, EXNREF:$rhs, I32:$cond),
                       (outs), (ins),
                       [(set EXNREF:$dst,
                         (select I32:$cond, EXNREF:$lhs, EXNREF:$rhs))],
                       "exnref.select\t$dst, $lhs, $rhs, $cond",
                       "exnref.select", 0x1b>;

def : Pat<(select (i32 (setne I32:$cond, 0)), EXNREF:$lhs, EXNREF:$rhs),
          (SELECT_EXNREF EXNREF:$lhs, EXNREF:$rhs, I32:$cond)>;
def : Pat<(select (i32 (seteq I32:$cond, 0)), EXNREF:$lhs, EXNREF:$rhs),
          (SELECT_EXNREF EXNREF:$rhs, EXNREF:$lhs, I32:$cond)>;
