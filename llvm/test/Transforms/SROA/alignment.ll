; RUN: opt < %s -sroa -S | FileCheck %s
; RUN: opt -debugify -sroa -S < %s | FileCheck %s -check-prefix DEBUGLOC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)

define void @test1({ i8, i8 }* %a, { i8, i8 }* %b) {
; CHECK-LABEL: @test1(
; CHECK: %[[gep_a0:.*]] = getelementptr inbounds { i8, i8 }, { i8, i8 }* %a, i64 0, i32 0
; CHECK: %[[a0:.*]] = load i8, i8* %[[gep_a0]], align 16
; CHECK: %[[gep_a1:.*]] = getelementptr inbounds { i8, i8 }, { i8, i8 }* %a, i64 0, i32 1
; CHECK: %[[a1:.*]] = load i8, i8* %[[gep_a1]], align 1
; CHECK: %[[gep_b0:.*]] = getelementptr inbounds { i8, i8 }, { i8, i8 }* %b, i64 0, i32 0
; CHECK: store i8 %[[a0]], i8* %[[gep_b0]], align 16
; CHECK: %[[gep_b1:.*]] = getelementptr inbounds { i8, i8 }, { i8, i8 }* %b, i64 0, i32 1
; CHECK: store i8 %[[a1]], i8* %[[gep_b1]], align 1
; CHECK: ret void

entry:
  %alloca = alloca { i8, i8 }, align 16
  %gep_a = getelementptr { i8, i8 }, { i8, i8 }* %a, i32 0, i32 0
  %gep_alloca = getelementptr { i8, i8 }, { i8, i8 }* %alloca, i32 0, i32 0
  %gep_b = getelementptr { i8, i8 }, { i8, i8 }* %b, i32 0, i32 0

  store i8 420, i8* %gep_alloca, align 16

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %gep_alloca, i8* align 16 %gep_a, i32 2, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 16 %gep_b, i8* align 16 %gep_alloca, i32 2, i1 false)
  ret void
}

define void @test2() {
; CHECK-LABEL: @test2(
; CHECK: alloca i16
; CHECK: load i8, i8* %{{.*}}
; CHECK: store i8 42, i8* %{{.*}}
; CHECK: ret void

; Check that when sroa rewrites the alloca partition
; it preserves the original DebugLocation.
; DEBUGLOC-LABEL: @test2(
; DEBUGLOC: {{.*}} = alloca {{.*}} !dbg ![[DbgLoc:[0-9]+]]
; DEBUGLOC-LABEL: }
;
; DEBUGLOC: ![[DbgLoc]] = !DILocation(line: 9,

entry:
  %a = alloca { i8, i8, i8, i8 }, align 2      ; "line 9" to -debugify
  %gep1 = getelementptr { i8, i8, i8, i8 }, { i8, i8, i8, i8 }* %a, i32 0, i32 1
  %cast1 = bitcast i8* %gep1 to i16*
  store volatile i16 0, i16* %cast1
  %gep2 = getelementptr { i8, i8, i8, i8 }, { i8, i8, i8, i8 }* %a, i32 0, i32 2
  %result = load i8, i8* %gep2
  store i8 42, i8* %gep2
  ret void
}

define void @PR13920(<2 x i64>* %a, i16* %b) {
; Test that alignments on memcpy intrinsics get propagated to loads and stores.
; CHECK-LABEL: @PR13920(
; CHECK: load <2 x i64>, <2 x i64>* %a, align 2
; CHECK: store <2 x i64> {{.*}}, <2 x i64>* {{.*}}, align 2
; CHECK: ret void

entry:
  %aa = alloca <2 x i64>, align 16
  %aptr = bitcast <2 x i64>* %a to i8*
  %aaptr = bitcast <2 x i64>* %aa to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %aaptr, i8* align 2 %aptr, i32 16, i1 false)
  %bptr = bitcast i16* %b to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %bptr, i8* align 2 %aaptr, i32 16, i1 false)
  ret void
}

define void @test3(i8* %x) {
; Test that when we promote an alloca to a type with lower ABI alignment, we
; provide the needed explicit alignment that code using the alloca may be
; expecting. However, also check that any offset within an alloca can in turn
; reduce the alignment.
; CHECK-LABEL: @test3(
; CHECK: alloca [22 x i8], align 8
; CHECK: alloca [18 x i8], align 2
; CHECK: ret void

entry:
  %a = alloca { i8*, i8*, i8* }
  %b = alloca { i8*, i8*, i8* }
  %a_raw = bitcast { i8*, i8*, i8* }* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %a_raw, i8* align 8 %x, i32 22, i1 false)
  %b_raw = bitcast { i8*, i8*, i8* }* %b to i8*
  %b_gep = getelementptr i8, i8* %b_raw, i32 6
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %b_gep, i8* align 2 %x, i32 18, i1 false)
  ret void
}

define void @test5() {
; Test that we preserve underaligned loads and stores when splitting. The use
; of volatile in this test case is just to force the loads and stores to not be
; split or promoted out of existence.
;
; CHECK-LABEL: @test5(
; CHECK: alloca [9 x i8]
; CHECK: alloca [9 x i8]
; CHECK: store volatile double 0.0{{.*}}, double* %{{.*}}, align 1
; CHECK: load volatile i16, i16* %{{.*}}, align 1
; CHECK: load double, double* %{{.*}}, align 1
; CHECK: store volatile double %{{.*}}, double* %{{.*}}, align 1
; CHECK: load volatile i16, i16* %{{.*}}, align 1
; CHECK: ret void

entry:
  %a = alloca [18 x i8]
  %raw1 = getelementptr inbounds [18 x i8], [18 x i8]* %a, i32 0, i32 0
  %ptr1 = bitcast i8* %raw1 to double*
  store volatile double 0.0, double* %ptr1, align 1
  %weird_gep1 = getelementptr inbounds [18 x i8], [18 x i8]* %a, i32 0, i32 7
  %weird_cast1 = bitcast i8* %weird_gep1 to i16*
  %weird_load1 = load volatile i16, i16* %weird_cast1, align 1

  %raw2 = getelementptr inbounds [18 x i8], [18 x i8]* %a, i32 0, i32 9
  %ptr2 = bitcast i8* %raw2 to double*
  %d1 = load double, double* %ptr1, align 1
  store volatile double %d1, double* %ptr2, align 1
  %weird_gep2 = getelementptr inbounds [18 x i8], [18 x i8]* %a, i32 0, i32 16
  %weird_cast2 = bitcast i8* %weird_gep2 to i16*
  %weird_load2 = load volatile i16, i16* %weird_cast2, align 1

  ret void
}

define void @test6() {
; Test that we promote alignment when the underlying alloca switches to one
; that innately provides it.
; CHECK-LABEL: @test6(
; CHECK: alloca double
; CHECK: alloca double
; CHECK-NOT: align
; CHECK: ret void

entry:
  %a = alloca [16 x i8]
  %raw1 = getelementptr inbounds [16 x i8], [16 x i8]* %a, i32 0, i32 0
  %ptr1 = bitcast i8* %raw1 to double*
  store volatile double 0.0, double* %ptr1, align 1

  %raw2 = getelementptr inbounds [16 x i8], [16 x i8]* %a, i32 0, i32 8
  %ptr2 = bitcast i8* %raw2 to double*
  %val = load double, double* %ptr1, align 1
  store volatile double %val, double* %ptr2, align 1

  ret void
}

define void @test7(i8* %out) {
; Test that we properly compute the destination alignment when rewriting
; memcpys as direct loads or stores.
; CHECK-LABEL: @test7(
; CHECK-NOT: alloca

entry:
  %a = alloca [16 x i8]
  %raw1 = getelementptr inbounds [16 x i8], [16 x i8]* %a, i32 0, i32 0
  %ptr1 = bitcast i8* %raw1 to double*
  %raw2 = getelementptr inbounds [16 x i8], [16 x i8]* %a, i32 0, i32 8
  %ptr2 = bitcast i8* %raw2 to double*

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %raw1, i8* %out, i32 16, i1 false)
; CHECK: %[[val2:.*]] = load double, double* %{{.*}}, align 1
; CHECK: %[[val1:.*]] = load double, double* %{{.*}}, align 1

  %val1 = load double, double* %ptr2, align 1
  %val2 = load double, double* %ptr1, align 1

  store double %val1, double* %ptr1, align 1
  store double %val2, double* %ptr2, align 1

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %out, i8* %raw1, i32 16, i1 false)
; CHECK: store double %[[val1]], double* %{{.*}}, align 1
; CHECK: store double %[[val2]], double* %{{.*}}, align 1

  ret void
; CHECK: ret void
}

define void @test8() {
; CHECK-LABEL: @test8(
; CHECK: load i32, {{.*}}, align 1
; CHECK: load i32, {{.*}}, align 1
; CHECK: load i32, {{.*}}, align 1
; CHECK: load i32, {{.*}}, align 1
; CHECK: load i32, {{.*}}, align 1

  %ptr = alloca [5 x i32], align 1
  %ptr.8 = bitcast [5 x i32]* %ptr to i8*
  call void @populate(i8* %ptr.8)
  %val = load [5 x i32], [5 x i32]* %ptr, align 1
  ret void
}

define void @test9() {
; CHECK-LABEL: @test9(
; CHECK: load i32, {{.*}}, align 8
; CHECK: load i32, {{.*}}, align 4
; CHECK: load i32, {{.*}}, align 8
; CHECK: load i32, {{.*}}, align 4
; CHECK: load i32, {{.*}}, align 8

  %ptr = alloca [5 x i32], align 8
  %ptr.8 = bitcast [5 x i32]* %ptr to i8*
  call void @populate(i8* %ptr.8)
  %val = load [5 x i32], [5 x i32]* %ptr, align 8
  ret void
}

define void @test10() {
; CHECK-LABEL: @test10(
; CHECK: load i32, {{.*}}, align 2
; CHECK: load i8, {{.*}}, align 2
; CHECK: load i8, {{.*}}, align 1
; CHECK: load i8, {{.*}}, align 2
; CHECK: load i16, {{.*}}, align 2

  %ptr = alloca {i32, i8, i8, {i8, i16}}, align 2
  %ptr.8 = bitcast {i32, i8, i8, {i8, i16}}* %ptr to i8*
  call void @populate(i8* %ptr.8)
  %val = load {i32, i8, i8, {i8, i16}}, {i32, i8, i8, {i8, i16}}* %ptr, align 2
  ret void
}

%struct = type { i32, i32 }
define dso_local i32 @pr45010(%struct* %A) {
; CHECK-LABEL: @pr45010
; CHECK: load atomic volatile i32, {{.*}}, align 4

  %B = alloca %struct, align 4
  %A.i = getelementptr inbounds %struct, %struct* %A, i32 0, i32 0
  %B.i = getelementptr inbounds %struct, %struct* %B, i32 0, i32 0
  %1 = load i32, i32* %A.i, align 4
  store atomic volatile i32 %1, i32* %B.i release, align 4
  %2 = bitcast %struct* %B to i32*
  %x = load atomic volatile i32, i32* %2 acquire, align 4
  ret i32 %x
}

declare void @populate(i8*)
