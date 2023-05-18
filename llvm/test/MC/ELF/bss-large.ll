; RUN: llc -filetype=obj %s -o %t

; PR16338 - ICE when compiling very large two-dimensional array
; Check if a huge object can be put into bss section
; C++ code is:
;   int a[60666][60666];

; ModuleID = 'test.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a0 = addrspace(1) global [4 x [4 x i32]] zeroinitializer, align 16
@a = global [60666 x [60666 x i32]] zeroinitializer, align 16
