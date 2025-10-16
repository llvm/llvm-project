; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s
; RUN: opt -passes='global-merge<max-offset=100>' -S -o - %s | FileCheck %s

; NOTE: This is a copy of the llvm/test/Transforms/GlobalMerge/basic.ll test,
; using `private` global variables instead of `internal`. This is to show that
; that private globals can be merged in the GlobalMerge pass.

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @_MergedGlobals = private global <{ i32, i32 }> <{ i32 1, i32 2 }>, align 4
; CHECK: @_MergedGlobals.1 = private global <{ i32, i32 }> <{ i32 3, i32 4 }>, section "foo", align 4

; CHECK-DAG: @a = private alias i32, ptr @_MergedGlobals{{$}}
@a = private global i32 1

; CHECK-DAG: @b = private alias i32, getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 1)
@b = private global i32 2

; CHECK-DAG: @c = private alias i32, ptr @_MergedGlobals.1{{$}}
@c = private global i32 3, section "foo"

; CHECK-DAG: @d = private alias i32, getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals.1, i32 0, i32 1)
@d = private global i32 4, section "foo"

define void @use_private() {
  ; CHECK: load i32, ptr @_MergedGlobals,
  %x = load i32, ptr @a
  ; CHECK: load i32, ptr getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 1)
  %y = load i32, ptr @b
  ; CHECK: load i32, ptr @_MergedGlobals.1
  %z1 = load i32, ptr @c
  ; CHECK: load i32, ptr getelementptr inbounds (<{ i32, i32 }>, ptr @_MergedGlobals.1, i32 0, i32 1)
  %z2 = load i32, ptr @d
  ret void
}
