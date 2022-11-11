; RUN: opt < %s -passes=asan -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-mapping-scale=5 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

; Here we check that the global redzone sizes grow with the object size.

@G10 = global [10 x i8] zeroinitializer, align 1
; CHECK: @G10 = global { [10 x i8], [22 x i8] }

@G31 = global [31 x i8] zeroinitializer, align 1
@G32 = global [32 x i8] zeroinitializer, align 1
@G33 = global [33 x i8] zeroinitializer, align 1
; CHECK: @G31 = global { [31 x i8], [33 x i8] }
; CHECK: @G32 = global { [32 x i8], [32 x i8] }
; CHECK: @G33 = global { [33 x i8], [63 x i8] }

@G63 = global [63 x i8] zeroinitializer, align 1
@G64 = global [64 x i8] zeroinitializer, align 1
@G65 = global [65 x i8] zeroinitializer, align 1
; CHECK: @G63 = global { [63 x i8], [33 x i8] }
; CHECK: @G64 = global { [64 x i8], [32 x i8] }
; CHECK: @G65 = global { [65 x i8], [63 x i8] }

@G127 = global [127 x i8] zeroinitializer, align 1
@G128 = global [128 x i8] zeroinitializer, align 1
@G129 = global [129 x i8] zeroinitializer, align 1
; CHECK: @G127 = global { [127 x i8], [33 x i8] }
; CHECK: @G128 = global { [128 x i8], [32 x i8] }
; CHECK: @G129 = global { [129 x i8], [63 x i8] }

@G255 = global [255 x i8] zeroinitializer, align 1
@G256 = global [256 x i8] zeroinitializer, align 1
@G257 = global [257 x i8] zeroinitializer, align 1
; CHECK: @G255 = global { [255 x i8], [33 x i8] }
; CHECK: @G256 = global { [256 x i8], [64 x i8] }
; CHECK: @G257 = global { [257 x i8], [95 x i8] }

@G511 = global [511 x i8] zeroinitializer, align 1
@G512 = global [512 x i8] zeroinitializer, align 1
@G513 = global [513 x i8] zeroinitializer, align 1
; CHECK: @G511 = global { [511 x i8], [97 x i8] }
; CHECK: @G512 = global { [512 x i8], [128 x i8] }
; CHECK: @G513 = global { [513 x i8], [159 x i8] }

@G1023 = global [1023 x i8] zeroinitializer, align 1
@G1024 = global [1024 x i8] zeroinitializer, align 1
@G1025 = global [1025 x i8] zeroinitializer, align 1
; CHECK: @G1023 = global { [1023 x i8], [225 x i8] }
; CHECK: @G1024 = global { [1024 x i8], [256 x i8] }
; CHECK: @G1025 = global { [1025 x i8], [287 x i8] }

@G1000000 = global [1000000 x i8] zeroinitializer, align 1
@G10000000 = global [10000000 x i8] zeroinitializer, align 1
@G100000000 = global [100000000 x i8] zeroinitializer, align 1
; CHECK: @G1000000 = global { [1000000 x i8], [249984 x i8] }
; CHECK: @G10000000 = global { [10000000 x i8], [262144 x i8] }
; CHECK: @G100000000 = global { [100000000 x i8], [262144 x i8] }
