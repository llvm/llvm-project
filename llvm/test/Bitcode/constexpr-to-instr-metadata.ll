; RUN: llvm-dis -expand-constant-exprs < %S/Inputs/constexpr-to-instr-metadata.bc | FileCheck %s

; CHECK-LABEL: define void @test() {
; CHECK: %constexpr = ptrtoint ptr @g to i32
; CHECK: %constexpr1 = zext i32 %constexpr to i64
; CHECK: %constexpr2 = ptrtoint ptr @g to i64
; CHECK: %constexpr3 = lshr i64 %constexpr2, 32
; CHECK: %constexpr4 = trunc i64 %constexpr3 to i32
; CHECK: %constexpr5 = zext i32 %constexpr4 to i64
; CHECK: %constexpr6 = shl i64 %constexpr5, 32
; CHECK: %constexpr7 = or i64 %constexpr1, %constexpr6
; CHECK: call void @llvm.dbg.value(metadata i64 %constexpr7, metadata !4, metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg !13
