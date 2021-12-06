// WebAssemblyInstrRef.td - WebAssembly reference type codegen --*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly reference type operand codegen constructs.
///
//===----------------------------------------------------------------------===//

multiclass REF_I<WebAssemblyRegClass rc, ValueType vt, string ht> {
  defm REF_NULL_#rc : I<(outs rc:$dst), (ins),
                        (outs), (ins),
                        [(set rc:$dst, (!cast<Intrinsic>("int_wasm_ref_null_" # ht)))],
                        "ref.null_" # ht # "$dst",
                        "ref.null_" # ht,
                        !cond(!eq(ht, "func")   : 0xd070, 
                              !eq(ht, "extern") : 0xd06f)>,
                      Requires<[HasReferenceTypes]>;
  defm SELECT_#rc: I<(outs rc:$dst), (ins rc:$lhs, rc:$rhs, I32:$cond),
                     (outs), (ins),
                     [(set rc:$dst,
                       (select I32:$cond, rc:$lhs, rc:$rhs))],
                     vt#".select\t$dst, $lhs, $rhs, $cond",
                     vt#".select", 0x1b>,
                   Requires<[HasReferenceTypes]>;
}

defm "" : REF_I<FUNCREF, funcref, "func">;
defm "" : REF_I<EXTERNREF, externref, "extern">;

foreach rc = [FUNCREF, EXTERNREF] in {
def : Pat<(select (i32 (setne I32:$cond, 0)), rc:$lhs, rc:$rhs),
          (!cast<Instruction>("SELECT_"#rc) rc:$lhs, rc:$rhs, I32:$cond)>;
def : Pat<(select (i32 (seteq I32:$cond, 0)), rc:$lhs, rc:$rhs),
          (!cast<Instruction>("SELECT_"#rc) rc:$rhs, rc:$lhs, I32:$cond)>;
}
