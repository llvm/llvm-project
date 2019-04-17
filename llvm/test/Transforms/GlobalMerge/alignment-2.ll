; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; This produces align 4, not the obvious align 1, to be consistent with what
; the AsmPrinter would do.
; CHECK: @_MergedGlobals = private global <{ [2 x i32], [2 x i32] }> <{ [2 x i32] [i32 1, i32 1], [2 x i32] [i32 2, i32 2] }>, align 4

; CHECK: @a = internal alias [2 x i32], getelementptr inbounds (<{ [2 x i32], [2 x i32] }>, <{ [2 x i32], [2 x i32] }>* @_MergedGlobals, i32 0, i32 0)
@a = internal global [2 x i32] [i32 1, i32 1], align 1

; CHECK: @b = internal alias [2 x i32], getelementptr inbounds (<{ [2 x i32], [2 x i32] }>, <{ [2 x i32], [2 x i32] }>* @_MergedGlobals, i32 0, i32 1)
@b = internal global [2 x i32] [i32 2, i32 2], align 1

define void @use() {
  ; CHECK: load i32, i32* getelementptr inbounds (<{ [2 x i32], [2 x i32] }>, <{ [2 x i32], [2 x i32] }>* @_MergedGlobals, i32 0, i32 0, i32 0)
  %x = load i32, i32* bitcast ([2 x i32]* @a to i32*)
  ; CHECK: load i32, i32* getelementptr inbounds (<{ [2 x i32], [2 x i32] }>, <{ [2 x i32], [2 x i32] }>* @_MergedGlobals, i32 0, i32 1, i32 0)
  %y = load i32, i32* bitcast ([2 x i32]* @b to i32*)
  ret void
}
