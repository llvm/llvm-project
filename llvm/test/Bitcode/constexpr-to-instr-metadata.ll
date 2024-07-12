; RUN: llvm-dis -expand-constant-exprs < %S/Inputs/constexpr-to-instr-metadata.bc | FileCheck %s

; CHECK-LABEL: define void @test() {
; CHECK: #dbg_value(i64 undef, !4, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !13
