; RUN: opt -debugify -dce -S < %s | FileCheck %s
; RUN: opt -passes='module(debugify),function(dce)' -S < %s | FileCheck %s

; CHECK-LABEL: @test
define void @test() {
  %add = add i32 1, 2
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 1, metadata [[add:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2, DW_OP_stack_value))
  %sub = sub i32 %add, 1
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 1, metadata [[sub:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2, DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value))
; CHECK-NEXT: ret void
  ret void
}

; CHECK: [[add]] = !DILocalVariable
; CHECK: [[sub]] = !DILocalVariable
