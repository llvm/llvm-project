; RUN: llvm-dis -expand-constant-exprs < %S/Inputs/constexpr-to-instr-metadata.bc | FileCheck %s

; CHECK-LABEL: define void @test() {
; CHECK: call void @llvm.dbg.value(metadata i64 undef, metadata !4, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !13
