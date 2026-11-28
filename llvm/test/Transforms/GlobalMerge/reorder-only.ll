; Verify -global-merge-reorder-only: by default globals are merged; with the
; option they are only placed next to each other, keeping their own symbols.

; RUN: opt -global-merge -global-merge-max-offset=200 -S %s | FileCheck %s --check-prefix=MERGE
; RUN: opt -global-merge -global-merge-max-offset=200 -global-merge-reorder-only -S %s | FileCheck %s --check-prefix=REORDER

target triple = "arm64-apple-macos"

; MERGE: @_MergedGlobals

; Merge order is by size, so the globals are laid out small to large.
; REORDER: @s = internal global i16
; REORDER-NEXT: @a = internal global i32
; REORDER-NEXT: @b = internal global i32
; REORDER-NEXT: @big = internal global [100 x i8]
; REORDER-NOT: @_MergedGlobals

@big = internal global [100 x i8] zeroinitializer
@a = internal global i32 1
@s = internal global i16 3
@b = internal global i32 2

define void @use() {
  %x = load i32, ptr @a
  %y = load i32, ptr @b
  %z = load i16, ptr @s
  %w = load i8, ptr @big
  ret void
}
