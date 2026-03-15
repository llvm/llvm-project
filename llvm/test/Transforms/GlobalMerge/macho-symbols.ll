; RUN: opt -global-merge -global-merge-max-offset=100 -S -o - %s | FileCheck %s
; RUN: opt -passes='global-merge<max-offset=100>' -S -o - %s | FileCheck %s

;; For Mach-O, we do not expect any alias symbols to be created for
;; internal/private symbols by GlobalMerge.

target datalayout = "e-p:64:64"
target triple = "x86_64-apple-macos11"

@a = private global i32 1
@b = private global i32 2
@c = internal global i32 3
@d = internal global i32 4

; CHECK: @_MergedGlobals = internal global <{ i32, i32, i32, i32 }> <{ i32 1, i32 2, i32 3, i32 4 }>, align 4
; CHECK-NOT: alias

define void @use() {
  ; CHECK: load i32, ptr @_MergedGlobals,
  %x = load i32, ptr @a
  ; CHECK: load i32, ptr getelementptr inbounds (<{ i32, i32, i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 1)
  %y = load i32, ptr @b
  ; CHECK: load i32, ptr getelementptr inbounds (<{ i32, i32, i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 2)
  %z1 = load i32, ptr @c
  ; CHECK: load i32, ptr getelementptr inbounds (<{ i32, i32, i32, i32 }>, ptr @_MergedGlobals, i32 0, i32 3)
  %z2 = load i32, ptr @d
  ret void
}
