; RUN: opt -passes='module(debugify),function(dce)' -S < %s | FileCheck %s

; CHECK-LABEL: @test
define void @test() {
  %add = add i32 1, 2
; CHECK-NEXT: #dbg_value(i32 1, [[add:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 2, DW_OP_stack_value),
  %sub = sub i32 %add, 1
; CHECK-NEXT: #dbg_value(i32 1, [[sub:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 2, DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value),
; CHECK-NEXT: ret void
  ret void
}

declare void @llvm.lifetime.start.p0(ptr nocapture) nounwind
declare void @llvm.lifetime.end.p0(ptr nocapture) nounwind

; CHECK-LABEL: @test_lifetime_alloca
define i32 @test_lifetime_alloca() {
; Check that lifetime intrinsics are removed along with the pointer.
; CHECK-NEXT: #dbg_value
; CHECK-NEXT: ret i32 0
; CHECK-NOT: llvm.lifetime.start
; CHECK-NOT: llvm.lifetime.end
  %i = alloca i8, align 4
  call void @llvm.lifetime.start.p0(ptr %i)
  call void @llvm.lifetime.end.p0(ptr %i)
  ret i32 0
}

; CHECK: [[add]] = !DILocalVariable
; CHECK: [[sub]] = !DILocalVariable
