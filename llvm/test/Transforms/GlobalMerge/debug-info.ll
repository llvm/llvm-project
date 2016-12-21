; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @_MergedGlobals = private global { i32, i32 } { i32 1, i32 2 }, !dbg [[A:![0-9]+]], !dbg [[B:![0-9]+]]

@a = internal global i32 1, !dbg !0
@b = internal global i32 2, !dbg !1

define void @use1() {
  %x = load i32, i32* @a
  %y = load i32, i32* @b
  ret void
}

; CHECK: [[A]] = !DIGlobalVariableExpression(var: [[AVAR:![0-9]+]])
; CHECK: [[AVAR]] = !DIGlobalVariable(name: "a", scope: null, isLocal: false, isDefinition: true)
; CHECK: [[B]] = !DIGlobalVariableExpression(var: [[BVAR:![0-9]+]], expr: [[EXPR:![0-9]+]])
; CHECK: [[BVAR]] = !DIGlobalVariable(name: "b", scope: null, isLocal: false, isDefinition: true)
; CHECK: [[EXPR]] = !DIExpression(DW_OP_plus, 4)

!llvm.module.flags = !{!2, !3}
!0 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "a"))
!1 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "b"))
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 4}
