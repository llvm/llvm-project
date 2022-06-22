; RUN: opt < %s -passes=sroa -S | FileCheck %s
; RUN: opt -passes='debugify,function(sroa)' -S < %s | FileCheck %s -check-prefix DEBUGLOC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)

define void @test1(ptr %a, ptr %b) {
; CHECK-LABEL: @test1(
; CHECK: %[[gep_a0:.*]] = getelementptr { i8, i8 }, ptr %a, i32 0, i32 0
; CHECK: %[[gep_b0:.*]] = getelementptr { i8, i8 }, ptr %b, i32 0, i32 0
; CHECK: %[[a0:.*]] = load i8, ptr %[[gep_a0]], align 16
; CHECK: %[[gep_a1:.*]] = getelementptr inbounds i8, ptr %[[gep_a0]], i64 1
; CHECK: %[[a1:.*]] = load i8, ptr %[[gep_a1]], align 1
; CHECK: store i8 %[[a0]], ptr %[[gep_b0]], align 16
; CHECK: %[[gep_b1:.*]] = getelementptr inbounds i8, ptr %[[gep_b0]], i64 1
; CHECK: store i8 %[[a1]], ptr %[[gep_b1]], align 1
; CHECK: ret void

entry:
  %alloca = alloca { i8, i8 }, align 16
  %gep_a = getelementptr { i8, i8 }, ptr %a, i32 0, i32 0
  %gep_alloca = getelementptr { i8, i8 }, ptr %alloca, i32 0, i32 0
  %gep_b = getelementptr { i8, i8 }, ptr %b, i32 0, i32 0

  store i8 420, ptr %gep_alloca, align 16

  call void @llvm.memcpy.p0.p0.i32(ptr align 16 %gep_alloca, ptr align 16 %gep_a, i32 2, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 16 %gep_b, ptr align 16 %gep_alloca, i32 2, i1 false)
  ret void
}

define void @test2() {
; CHECK-LABEL: @test2(
; CHECK: alloca i16
; CHECK: load i8, ptr %{{.*}}
; CHECK: store i8 42, ptr %{{.*}}
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
  %gep1 = getelementptr { i8, i8, i8, i8 }, ptr %a, i32 0, i32 1
  store volatile i16 0, ptr %gep1
  %gep2 = getelementptr { i8, i8, i8, i8 }, ptr %a, i32 0, i32 2
  %result = load i8, ptr %gep2
  store i8 42, ptr %gep2
  ret void
}

define void @PR13920(ptr %a, ptr %b) {
; Test that alignments on memcpy intrinsics get propagated to loads and stores.
; CHECK-LABEL: @PR13920(
; CHECK: load <2 x i64>, ptr %a, align 2
; CHECK: store <2 x i64> {{.*}}, ptr {{.*}}, align 2
; CHECK: ret void

entry:
  %aa = alloca <2 x i64>, align 16
  call void @llvm.memcpy.p0.p0.i32(ptr align 2 %aa, ptr align 2 %a, i32 16, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 2 %b, ptr align 2 %aa, i32 16, i1 false)
  ret void
}

define void @test3(ptr %x) {
; Test that when we promote an alloca to a type with lower ABI alignment, we
; provide the needed explicit alignment that code using the alloca may be
; expecting. However, also check that any offset within an alloca can in turn
; reduce the alignment.
; CHECK-LABEL: @test3(
; CHECK: alloca [22 x i8], align 8
; CHECK: alloca [18 x i8], align 2
; CHECK: ret void

entry:
  %a = alloca { ptr, ptr, ptr }
  %b = alloca { ptr, ptr, ptr }
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %a, ptr align 8 %x, i32 22, i1 false)
  %b_gep = getelementptr i8, ptr %b, i32 6
  call void @llvm.memcpy.p0.p0.i32(ptr align 2 %b_gep, ptr align 2 %x, i32 18, i1 false)
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
; CHECK: store volatile double 0.0{{.*}}, ptr %{{.*}}, align 1
; CHECK: load volatile i16, ptr %{{.*}}, align 1
; CHECK: load double, ptr %{{.*}}, align 1
; CHECK: store volatile double %{{.*}}, ptr %{{.*}}, align 1
; CHECK: load volatile i16, ptr %{{.*}}, align 1
; CHECK: ret void

entry:
  %a = alloca [18 x i8]
  store volatile double 0.0, ptr %a, align 1
  %weird_gep1 = getelementptr inbounds [18 x i8], ptr %a, i32 0, i32 7
  %weird_load1 = load volatile i16, ptr %weird_gep1, align 1

  %raw2 = getelementptr inbounds [18 x i8], ptr %a, i32 0, i32 9
  %d1 = load double, ptr %a, align 1
  store volatile double %d1, ptr %raw2, align 1
  %weird_gep2 = getelementptr inbounds [18 x i8], ptr %a, i32 0, i32 16
  %weird_load2 = load volatile i16, ptr %weird_gep2, align 1

  ret void
}

define void @test6() {
; We should set the alignment on all load and store operations; make sure
; we choose an appropriate alignment.
; CHECK-LABEL: @test6(
; CHECK: alloca double, align 8{{$}}
; CHECK: alloca double, align 8{{$}}
; CHECK: store{{.*}}, align 8
; CHECK: load{{.*}}, align 8
; CHECK: store{{.*}}, align 8
; CHECK-NOT: align
; CHECK: ret void

entry:
  %a = alloca [16 x i8]
  store volatile double 0.0, ptr %a, align 1

  %raw2 = getelementptr inbounds [16 x i8], ptr %a, i32 0, i32 8
  %val = load double, ptr %a, align 1
  store volatile double %val, ptr %raw2, align 1

  ret void
}

define void @test7(ptr %out) {
; Test that we properly compute the destination alignment when rewriting
; memcpys as direct loads or stores.
; CHECK-LABEL: @test7(
; CHECK-NOT: alloca

entry:
  %a = alloca [16 x i8]
  %raw2 = getelementptr inbounds [16 x i8], ptr %a, i32 0, i32 8

  call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %out, i32 16, i1 false)
; CHECK: %[[val2:.*]] = load double, ptr %{{.*}}, align 1
; CHECK: %[[val1:.*]] = load double, ptr %{{.*}}, align 1

  %val1 = load double, ptr %raw2, align 1
  %val2 = load double, ptr %a, align 1

  store double %val1, ptr %a, align 1
  store double %val2, ptr %raw2, align 1

  call void @llvm.memcpy.p0.p0.i32(ptr %out, ptr %a, i32 16, i1 false)
; CHECK: store double %[[val1]], ptr %{{.*}}, align 1
; CHECK: store double %[[val2]], ptr %{{.*}}, align 1

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
  call void @populate(ptr %ptr)
  %val = load [5 x i32], ptr %ptr, align 1
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
  call void @populate(ptr %ptr)
  %val = load [5 x i32], ptr %ptr, align 8
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
  call void @populate(ptr %ptr)
  %val = load {i32, i8, i8, {i8, i16}}, ptr %ptr, align 2
  ret void
}

%struct = type { i32, i32 }
define dso_local i32 @pr45010(ptr %A) {
; CHECK-LABEL: @pr45010
; CHECK: load atomic volatile i32, {{.*}}, align 4

  %B = alloca %struct, align 4
  %1 = load i32, ptr %A, align 4
  store atomic volatile i32 %1, ptr %B release, align 4
  %x = load atomic volatile i32, ptr %B acquire, align 4
  ret i32 %x
}

declare void @populate(ptr)
