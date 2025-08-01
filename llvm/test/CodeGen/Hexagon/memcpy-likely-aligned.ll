; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

%struct.e = type { i8, i8, [2 x i8] }
%struct.s = type { ptr }
%struct.o = type { %struct.n }
%struct.n = type { [2 x %struct.l] }
%struct.l = type { %struct.e, %struct.d, %struct.e }
%struct.d = type <{ i8, i8, i8, i8, [2 x i8], [2 x i8] }>

@y = global { <{ { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e }, { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } }> } { <{ { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e }, { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } }> <{ { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } { %struct.e { i8 3, i8 0, [2 x i8] undef }, { i8, i8, i8, [5 x i8] } { i8 -47, i8 2, i8 0, [5 x i8] undef }, %struct.e { i8 3, i8 0, [2 x i8] undef } }, { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } { %struct.e { i8 3, i8 0, [2 x i8] undef }, { i8, i8, i8, [5 x i8] } { i8 -47, i8 2, i8 0, [5 x i8] undef }, %struct.e { i8 3, i8 0, [2 x i8] undef } } }> }, align 4
@t = common global %struct.s zeroinitializer, align 4
@q = internal global ptr null, align 4

define void @foo() nounwind {
entry:
  %0 = load ptr, ptr @t, align 4
  store ptr %0, ptr @q, align 4
  %1 = load ptr, ptr @q, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %1, ptr align 4 @y, i32 32, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
