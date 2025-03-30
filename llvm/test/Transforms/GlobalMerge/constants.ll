; RUN: opt -global-merge -global-merge-max-offset=100 -global-merge-on-const -S < %s | FileCheck %s
; RUN: opt -global-merge -global-merge-max-offset=100 -global-merge-on-const -global-merge-all-const -S < %s | FileCheck %s --check-prefix=AGGRESSIVE
; RUN: opt -passes='global-merge<max-offset=100;merge-const>' -S < %s | FileCheck %s
; RUN: opt -passes='global-merge<max-offset=100;merge-const;merge-const-aggressive>' -S < %s | FileCheck %s --check-prefix=AGGRESSIVE

; CHECK: @_MergedGlobals = private constant <{ i32, i32 }> <{ i32 1, i32 2 }>, align 4
; AGGRESSIVE: @_MergedGlobals = private constant <{ i32, i32, i32 }> <{ i32 1, i32 2, i32 3 }>, align 4

@a = internal constant i32 1
@b = internal constant i32 2
@c = internal constant i32 3

define void @use() {
  %a = load i32, ptr @a
  %b = load i32, ptr @b
  ret void
}

define void @use2() {
  %c = load i32, ptr @c
  ret void
}
