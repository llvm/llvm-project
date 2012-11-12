; RUN: llc < %s -mtriple=thumb-apple-darwin | FileCheck %s
; Test the ARMGlobalMerge pass.  Use -march=thumb because it has a small
; value for the maximum offset (127).

; A local array that exceeds the maximum offset should not be merged.
; CHECK: g0:
@g0 = internal global [32 x i32] [ i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 1, i32 2 ]

; CHECK: _MergedGlobals:
@g1 = internal global i32 1
@g2 = internal global i32 2

; Make sure that the complete variable fits within the range of the maximum
; offset.  Having the starting offset in range is not sufficient.
; When this works properly, @g3 is placed in a separate chunk of merged globals.
; CHECK: _MergedGlobals1:
@g3 = internal global [30 x i32] [ i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10 ], align 4

; Global variables that can be placed in BSS should be kept together in a
; separate pool of merged globals.
; CHECK: _MergedGlobals2
@g4 = internal global i32 0
@g5 = internal global i32 0
