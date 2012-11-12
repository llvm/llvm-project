; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10.0.0"

; CHECK: align 3
@.v = linker_private unnamed_addr constant <4 x i32> <i32 1, i32 2, i32 3, i32 4>, align 8
; CHECK: align 2
@.strA = linker_private unnamed_addr constant [4 x i8] c"bar\00"
; CHECK-NOT: align
@.strB = linker_private unnamed_addr constant [4 x i8] c"foo\00", align 1
@.strC = linker_private unnamed_addr constant [4 x i8] c"baz\00", section "__TEXT,__cstring,cstring_literals", align 1
