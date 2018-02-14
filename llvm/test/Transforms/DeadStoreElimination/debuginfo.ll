; RUN: opt < %s -debugify -basicaa -dse -S | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

declare noalias i8* @malloc(i32)

declare void @test_f()

define i32* @test_salvage() {
; CHECK-LABEL: @test_salvage()
; CHECK-NEXT: malloc
; CHECK-NEXT: bitcast
; CHECK-NEXT: call void @test_f()
; CHECK-NEXT: store i32 0, i32* %P

; Check that all four original local variables have their values preserved.
; CHECK-NEXT: call void @llvm.dbg.value(metadata i8* %p, metadata !8, metadata !DIExpression()), !dbg !14
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32* %P, metadata !10, metadata !DIExpression()), !dbg !15
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32* %P, metadata !11, metadata !DIExpression(DW_OP_deref)), !dbg !18
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32* %P, metadata !13, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !19
  %p = tail call i8* @malloc(i32 4)
  %P = bitcast i8* %p to i32*
  %DEAD = load i32, i32* %P
  %DEAD2 = add i32 %DEAD, 1
  store i32 %DEAD2, i32* %P
  call void @test_f()
  store i32 0, i32* %P
  ret i32* %P
}

; CHECK: !8 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !9)
; CHECK: !10 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !9)
; CHECK: !11 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 3, type: !12)
; CHECK: !13 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 4, type: !12)
; CHECK-DAG: !14 = !DILocation(line: 1, column: 1, scope: !5)
; CHECK-DAG: !15 = !DILocation(line: 2, column: 1, scope: !5)
; CHECK-DAG: !18 = !DILocation(line: 3, column: 1, scope: !5)
; CHECK-DAG: !19 = !DILocation(line: 4, column: 1, scope: !5)
; CHECK-DAG: !16 = !DILocation(line: 6, column: 1, scope: !5)
; CHECK-DAG: !17 = !DILocation(line: 7, column: 1, scope: !5)
