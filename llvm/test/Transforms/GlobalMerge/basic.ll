; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @_MergedGlobals = private global { i32, i32 } { i32 1, i32 2 }

; CHECK: @a = internal alias i32, getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_MergedGlobals, i32 0, i32 0)
@a = internal global i32 1

; CHECK: @b = internal alias i32, getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_MergedGlobals, i32 0, i32 1)
@b = internal global i32 2

define void @use() {
  ; CHECK: load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_MergedGlobals, i32 0, i32 0)
  %x = load i32, i32* @a
  ; CHECK: load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_MergedGlobals, i32 0, i32 1)
  %y = load i32, i32* @b
  ret void
}
