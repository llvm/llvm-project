; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: __hexagon_memcpy_likely_aligned_min32bytes_mult8bytes

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"
target triple = "hexagon-unknown-linux-gnu"

%struct.e = type { i8, i8, [2 x i8] }
%struct.s = type { i8* }
%struct.o = type { %struct.n }
%struct.n = type { [2 x %struct.l] }
%struct.l = type { %struct.e, %struct.d, %struct.e }
%struct.d = type <{ i8, i8, i8, i8, [2 x i8], [2 x i8] }>

@y = global { <{ { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e }, { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } }> } { <{ { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e }, { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } }> <{ { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } { %struct.e { i8 3, i8 0, [2 x i8] undef }, { i8, i8, i8, [5 x i8] } { i8 -47, i8 2, i8 0, [5 x i8] undef }, %struct.e { i8 3, i8 0, [2 x i8] undef } }, { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } { %struct.e { i8 3, i8 0, [2 x i8] undef }, { i8, i8, i8, [5 x i8] } { i8 -47, i8 2, i8 0, [5 x i8] undef }, %struct.e { i8 3, i8 0, [2 x i8] undef } } }> }, align 4
@t = common global %struct.s zeroinitializer, align 4
@q = internal global %struct.o* null, align 4

define void @foo() nounwind {
entry:
  %0 = load i8*, i8** getelementptr inbounds (%struct.s, %struct.s* @t, i32 0, i32 0), align 4
  %1 = bitcast i8* %0 to %struct.o*
  store %struct.o* %1, %struct.o** @q, align 4
  %2 = load %struct.o*, %struct.o** @q, align 4
  %p = getelementptr inbounds %struct.o, %struct.o* %2, i32 0, i32 0
  %m = getelementptr inbounds %struct.n, %struct.n* %p, i32 0, i32 0
  %arraydecay = getelementptr inbounds [2 x %struct.l], [2 x %struct.l]* %m, i32 0, i32 0
  %3 = bitcast %struct.l* %arraydecay to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %3, i8* getelementptr inbounds ({ <{ { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e }, { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } }> }, { <{ { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e }, { %struct.e, { i8, i8, i8, [5 x i8] }, %struct.e } }> }* @y, i32 0, i32 0, i32 0, i32 0, i32 0), i32 32, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
