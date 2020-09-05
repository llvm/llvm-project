; RUN: opt < %s -debugify -basic-aa -dse -enable-dse-memoryssa=false -S | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

declare noalias i8* @malloc(i32)

declare void @test_f()

define i32* @test_salvage(i32 %arg) {
; Check that all four original local variables have their values preserved.
; CHECK-LABEL: @test_salvage(
; CHECK-NEXT: malloc
; CHECK-NEXT: @llvm.dbg.value(metadata i8* %p, metadata ![[p:.*]], metadata !DIExpression())
; CHECK-NEXT: bitcast
; CHECK-NEXT: @llvm.dbg.value(metadata i32* %P, metadata ![[P:.*]], metadata !DIExpression())
; CHECK-NEXT: @llvm.dbg.value(metadata i32 %arg, metadata ![[DEAD:.*]], metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value))
; CHECK-NEXT: call void @test_f()
; CHECK-NEXT: store i32 0, i32* %P

  %p = tail call i8* @malloc(i32 4)
  %P = bitcast i8* %p to i32*
  %DEAD = add i32 %arg, 1
  store i32 %DEAD, i32* %P
  call void @test_f()
  store i32 0, i32* %P
  ret i32* %P
}

; CHECK: ![[p]] = !DILocalVariable(name: "1"
; CHECK: ![[P]] = !DILocalVariable(name: "2"
; CHECK: ![[DEAD]] = !DILocalVariable(name: "3"
