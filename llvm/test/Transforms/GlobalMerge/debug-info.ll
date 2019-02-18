; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s

source_filename = "test/Transforms/GlobalMerge/debug-info.ll"
target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"
; CHECK: @_MergedGlobals = private global <{ i32, i32 }> <{ i32 1, i32 2 }>, align 4, !dbg [[A:![0-9]+]], !dbg [[B:![0-9]+]]

@a = internal global i32 1, !dbg !0
@b = internal global i32 2, !dbg !2

define void @use1() {
  %x = load i32, i32* @a
  %y = load i32, i32* @b
  ret void
}
; CHECK: [[A]] = !DIGlobalVariableExpression(var: [[AVAR:![0-9]+]], expr: !DIExpression())
; CHECK: [[AVAR]] = !DIGlobalVariable(name: "a", scope: null, type: !2, isLocal: false, isDefinition: true)
; CHECK: [[B]] = !DIGlobalVariableExpression(var: [[BVAR:![0-9]+]], expr: !DIExpression(DW_OP_plus_uconst, 4))
; CHECK: [[BVAR]] = !DIGlobalVariable(name: "b", scope: null, type: !2, isLocal: false, isDefinition: true)

!llvm.module.flags = !{!4, !5}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, type: !6, isLocal: false, isDefinition: true)
!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = !DIGlobalVariable(name: "b", scope: null, type: !6, isLocal: false, isDefinition: true)
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
