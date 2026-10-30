; air-emulate-f64: every double is its bits (i64) and its operations calls of
; the soft-f64 library (AIRSoftF64.inc), linked in as internal functions;
; constants are their bits, and nothing of double is left.
; RUN: opt -mtriple=air -passes=air-emulate-f64 -S %s \
; RUN:   | FileCheck %s --implicit-check-not=double
; The portable passes run it, before the AIR functions are written (the
; library's intrinsics become AIR's).
; RUN: llc -mtriple=air -air-portable %s -o - \
; RUN:   | FileCheck %s --check-prefix=PORTABLE --implicit-check-not=double

; CHECK: @g = addrspace(1) global <2 x i64> <i64 4609434218613702656, i64 4612811918334230528>
; CHECK: @s = addrspace(1) global { i32, [2 x i64] } { i32 1, [2 x i64] [i64 4602678819172646912, i64 -4616189618054758400] }
; CHECK: @table = addrspace(1) global [1 x ptr] [ptr @helper]

target triple = "air"

@g = addrspace(1) global <2 x double> <double 1.5, double 2.5>
@s = addrspace(1) global { i32, [2 x double] } { i32 1, [2 x double] [double 0.5, double -1.0] }
@table = addrspace(1) global [1 x ptr] [ptr @helper]

declare i32 @llvm.fptosi.sat.i32.f64(double)
declare double @llvm.maximumnum.f64(double, double)
declare double @llvm.fabs.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.fma.f64(double, double, double)
declare double @llvm.sqrt.f64(double)

; Arguments, returns, compares and selects of doubles.
; CHECK-LABEL: define internal i64 @helper(i64 %x, i64 %y)
; CHECK: %[[LT:[0-9]+]] = call i32 @__air_f64_lt(i64 %x, i64 %y)
; CHECK: %[[C:[0-9]+]] = icmp ne i32 %[[LT]], 0
; CHECK: %s = select i1 %[[C]], i64 %x, i64 4613937818241073152
; CHECK: ret i64
define internal double @helper(double %x, double %y) {
  %c = fcmp olt double %x, %y
  %s = select i1 %c, double %x, double 3.0
  %m = call double @llvm.maximumnum.f64(double %s, double 0.0)
  ret double %m
}

; CHECK-LABEL: define void @arith(
; CHECK: %v = load <2 x i64>, ptr addrspace(1) @g
; CHECK: call i64 @__air_f64_add(
; CHECK: call i64 @__air_f64_mul(
; CHECK: call i64 @__air_f64_div(
; CHECK: call i64 @__air_f64_fma(
; CHECK: call i64 @__air_f64_sqrt(
; CHECK: call i64 @__air_f64_floor(
; fneg and fabs on the sign bit
; CHECK: xor i64 %{{.*}}, -9223372036854775808
; CHECK: and i64 %{{.*}}, 9223372036854775807
; CHECK: %h = call i64 @helper(
; CHECK: call i64 @__air_f64_to_i64(
; CHECK: call i64 @llvm.smin.i64(
; CHECK: call i64 @llvm.smax.i64(
; CHECK: call i64 @__air_f64_from_f32(
; CHECK: call i32 @__air_f64_to_f32(
; CHECK: call i32 @__air_f64_to_f16_r({{.*}}, i32 1)
; CHECK: store <2 x i64>
; CHECK: ret void
define void @arith(ptr addrspace(1) %p, float %f) "air-kernel" {
  %v = load <2 x double>, ptr addrspace(1) @g
  %w = fadd <2 x double> %v, %v
  %a = extractelement <2 x double> %w, i64 0
  %b = extractelement <2 x double> %w, i64 1
  %m = fmul nnan double %a, %b
  %d = fdiv double %m, %a
  %fm = call double @llvm.fma.f64(double %d, double %a, double %b)
  %sq = call double @llvm.sqrt.f64(double %fm)
  %fl = call double @llvm.floor.f64(double %sq)
  %n = fneg double %fl
  %ab = call double @llvm.fabs.f64(double %n)
  %h = call double @helper(double %ab, double %b)
  %i = call i32 @llvm.fptosi.sat.i32.f64(double %h)
  store i32 %i, ptr addrspace(1) %p
  %x = fpext float %f to double
  %y = fadd double %x, %h
  %t = fptrunc double %y to float
  store float %t, ptr addrspace(1) %p
  %th = fptrunc double %y to half
  store half %th, ptr addrspace(1) %p
  %r = insertelement <2 x double> %w, double %y, i64 1
  store <2 x double> %r, ptr addrspace(1) %p
  ret void
}

; A double atomic add: a compare-exchange loop on the bits.
; CHECK-LABEL: define void @atomic(
; CHECK: %[[INIT:[0-9]+]] = load volatile i64, ptr addrspace(1) %q, align 8
; CHECK: loop.i:
; CHECK: %[[OLD:[0-9]+]] = phi i64 [ %[[INIT]], %{{.*}} ], [ %[[GOT:[0-9]+]], %loop.i ]
; CHECK: %[[NEW:[0-9]+]] = call i64 @__air_f64_add(i64 %[[OLD]], i64 4611686018427387904)
; CHECK: %[[CX:[0-9]+]] = cmpxchg ptr addrspace(1) %q, i64 %[[OLD]], i64 %[[NEW]] syncscope("agent") monotonic monotonic, align 8
; CHECK: %[[GOT]] = extractvalue { i64, i1 } %[[CX]], 0
; CHECK: atomicrmw xchg ptr addrspace(1) %q, i64 4607182418800017408 monotonic, align 8
define void @atomic(ptr addrspace(1) %q) "air-kernel" {
  %old = atomicrmw fadd ptr addrspace(1) %q, double 2.0 syncscope("agent") monotonic, align 8
  store double %old, ptr addrspace(1) %q
  %x = atomicrmw xchg ptr addrspace(1) %q, double 1.0 monotonic, align 8
  ret void
}

; Memory of doubles: allocas, GEPs, phis.
; CHECK-LABEL: define void @memory(
; CHECK: %a = alloca [4 x i64]
; CHECK: phi i64 [ 0, %entry ], [ %{{[0-9]+}}, %loop ]
; CHECK: getelementptr i64, ptr %a, i64 %i
define void @memory(ptr addrspace(1) %p, i64 %i) "air-kernel" {
entry:
  %a = alloca [4 x double]
  br label %loop
loop:
  %acc = phi double [ 0.0, %entry ], [ %next, %loop ]
  %e = getelementptr double, ptr %a, i64 %i
  store double %acc, ptr %e
  %next = fadd double %acc, 1.0
  %c = fcmp ult double %next, 4.0
  br i1 %c, label %loop, label %exit
exit:
  store double %next, ptr addrspace(1) %p
  ret void
}

; The library: internal, only the functions that are used.
; CHECK-LABEL: define internal i64 @__air_f64_add(
; CHECK-NOT: @__air_f64_add_r(

; PORTABLE-LABEL: define internal i64 @__air_f64_add(
; PORTABLE: call i32 @air.max.u.i32(
; PORTABLE-LABEL: define void @arith(
; PORTABLE: call i64 @__air_f64_add(
