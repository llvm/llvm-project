; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s
; RUN: opt -passes='global-merge<max-offset=100>' -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @_MergedGlobals = private global <{ i32, i32 }> <{ i32 3, i32 4 }>, section "foo", align 4
; CHECK: @_MergedGlobals.1 = private global <{ i32, i32 }> <{ i32 1, i32 2 }>, align 4

; CHECK-DAG: @a = internal alias i32, ptr @_MergedGlobals.1
@a = internal global i32 1

; CHECK-DAG: @b = internal alias i32, getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals.1, i32 0, i32 1)
@b = internal global i32 2

; CHECK-DAG: @c = internal alias i32, ptr @_MergedGlobals
@c = internal global i32 3, section "foo"

; CHECK-DAG: @d = internal alias i32, getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 1)
@d = internal global i32 4, section "foo"

define void @use() {
  ; CHECK: load i32, ptr @_MergedGlobals.1
  %x = load i32, ptr @a
  ; CHECK: load i32, ptr getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals.1, i32 0, i32 1)
  %y = load i32, ptr @b
  ; CHECK: load i32, ptr @_MergedGlobals
  %z1 = load i32, ptr @c
  ; CHECK: load i32, ptr getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 1)
  %z2 = load i32, ptr @d
  ret void
}
