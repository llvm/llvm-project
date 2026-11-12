; RUN: llvm-dis -expand-constant-exprs < %S/Inputs/constexpr-to-instr-metadata.bc | FileCheck %s

; CHECK-LABEL: define void @test() !guid !{{[0-9]+}} {
; CHECK: #dbg_value(i64 poison, !5, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !14
