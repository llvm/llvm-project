; RUN: llc -mtriple=hexagon -O3 -hexagon-align-global-arrays \
; RUN:   -stop-after=hexagon-global-array-alignment < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -O3 -stop-after=hexagon-global-array-alignment < %s \
; RUN:   | FileCheck %s --check-prefix=DISABLED

; At -O3 (no opt-for-size), the pass aligns integer arrays (including
; multi-dimensional ones) to at least 8 bytes, while non-array and non-integer
; globals are left untouched.

; CHECK: @int_array = global [4 x i32] zeroinitializer, align 8
; CHECK: @char_array = global [8 x i8] zeroinitializer, align 8
; CHECK: @multidim = global [3 x [5 x i32]] zeroinitializer, align 8
; CHECK: @explicit = global [4 x i32] zeroinitializer, align 16
; CHECK: @scalar = global i32 0, align 4

; Without the flag, the pass makes no changes.
; DISABLED: @int_array = global [4 x i32] zeroinitializer, align 4
; DISABLED: @char_array = global [8 x i8] zeroinitializer, align 1
; DISABLED: @multidim = global [3 x [5 x i32]] zeroinitializer, align 4
; DISABLED: @explicit = global [4 x i32] zeroinitializer, align 16
; DISABLED: @scalar = global i32 0, align 4

@int_array = global [4 x i32] zeroinitializer, align 4
@char_array = global [8 x i8] zeroinitializer, align 1
@multidim = global [3 x [5 x i32]] zeroinitializer, align 4
@explicit = global [4 x i32] zeroinitializer, align 16
@scalar = global i32 0, align 4
