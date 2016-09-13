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

; CHECK: [[A]] = distinct !DIGlobalVariable(name: "a", scope: null, isLocal: false, isDefinition: true)
; CHECK: [[B]] = distinct !DIGlobalVariable(name: "b", scope: null, isLocal: false, isDefinition: true, expr: [[EXPR:![0-9]+]])
; CHECK: [[EXPR]] = !DIExpression(DW_OP_plus, 4)


!0 = distinct !DIGlobalVariable(name: "a")
!1 = distinct !DIGlobalVariable(name: "b")
