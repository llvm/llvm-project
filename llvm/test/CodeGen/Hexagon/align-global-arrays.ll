; At -O3 the pass does not optimize for size, so every global integer array
; (including multi-dimensional ones) is aligned to at least 8 bytes.
; RUN: llc -mtriple=hexagon -O3 -hexagon-align-global-arrays \
; RUN:   -stop-after=hexagon-global-array-alignment < %s | FileCheck %s
;
; At -O2 the pass optimizes for size, so byte and half-word arrays with
; alignment <= 2 keep their native alignment, while word arrays are still
; aligned to 8 bytes.
; RUN: llc -mtriple=hexagon -O2 -hexagon-align-global-arrays \
; RUN:   -stop-after=hexagon-global-array-alignment < %s \
; RUN:   | FileCheck %s --check-prefix=OPTSIZE
;
; Without the flag the pass makes no changes.
; RUN: llc -mtriple=hexagon -O3 -stop-after=hexagon-global-array-alignment < %s \
; RUN:   | FileCheck %s --check-prefix=DISABLED

; CHECK: @int_array = global [4 x i32] zeroinitializer, align 8
; CHECK: @char_array = global [8 x i8] zeroinitializer, align 8
; CHECK: @short_array = global [4 x i16] zeroinitializer, align 8
; CHECK: @multidim = global [3 x [5 x i32]] zeroinitializer, align 8
; CHECK: @explicit = global [4 x i32] zeroinitializer, align 16
; CHECK: @scalar = global i32 0, align 4

; When optimizing for size, byte/half arrays at align <= 2 are left alone, but
; word arrays are still promoted to 8 bytes.
; OPTSIZE: @int_array = global [4 x i32] zeroinitializer, align 8
; OPTSIZE: @char_array = global [8 x i8] zeroinitializer, align 1
; OPTSIZE: @short_array = global [4 x i16] zeroinitializer, align 2
; OPTSIZE: @multidim = global [3 x [5 x i32]] zeroinitializer, align 8
; OPTSIZE: @explicit = global [4 x i32] zeroinitializer, align 16
; OPTSIZE: @scalar = global i32 0, align 4

; DISABLED: @int_array = global [4 x i32] zeroinitializer, align 4
; DISABLED: @char_array = global [8 x i8] zeroinitializer, align 1
; DISABLED: @short_array = global [4 x i16] zeroinitializer, align 2
; DISABLED: @multidim = global [3 x [5 x i32]] zeroinitializer, align 4
; DISABLED: @explicit = global [4 x i32] zeroinitializer, align 16
; DISABLED: @scalar = global i32 0, align 4

@int_array = global [4 x i32] zeroinitializer, align 4
@char_array = global [8 x i8] zeroinitializer, align 1
@short_array = global [4 x i16] zeroinitializer, align 2
@multidim = global [3 x [5 x i32]] zeroinitializer, align 4
@explicit = global [4 x i32] zeroinitializer, align 16
@scalar = global i32 0, align 4
