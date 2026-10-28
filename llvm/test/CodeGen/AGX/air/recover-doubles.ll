; With -agx-air-native-f64, agx-lower-air compiles calls to AIR's software f64 library (double as i64
; bits) rounded to nearest as double operations, which AGX lowers with its own
; software f64; a directed rounding stays a call.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -agx-air-native-f64 -S %t.air.ll | FileCheck %s
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -agx-air-native-f64=0 -S %t.air.ll \
; RUN:   | FileCheck %s --check-prefix=OFF

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: fadd double
; CHECK: fmul double
; CHECK: call double @llvm.fma.f64(
; CHECK: fcmp olt double
; CHECK: fptrunc double {{.*}} to float
; CHECK: fsub double
; CHECK-NOT: fdiv double
; CHECK: ret void
; OFF-NOT: fadd double
define void @k(ptr addrspace(1) %out, i64 %x, i64 %y) "air-kernel" {
  %a = call i64 @__air_f64_add(i64 %x, i64 %y)
  %m = call i64 @__air_f64_mul(i64 %a, i64 %y)
  %f = call i64 @__air_f64_fma(i64 %m, i64 %x, i64 %y)
  %l = call i32 @__air_f64_lt(i64 %f, i64 %x)
  %t = call i32 @__air_f64_to_f32_r(i64 %f, i32 1)
  %s = call i64 @__air_f64_sub_r(i64 %f, i64 %x, i32 1)
  %d = call i64 @__air_f64_div_r(i64 %s, i64 %x, i32 3)
  store volatile i64 %d, ptr addrspace(1) %out
  store volatile i32 %l, ptr addrspace(1) %out
  store volatile i32 %t, ptr addrspace(1) %out
  ret void
}

define internal i64 @__air_f64_add(i64 %a, i64 %b) { ret i64 %a }
define internal i64 @__air_f64_mul(i64 %a, i64 %b) { ret i64 %a }
define internal i64 @__air_f64_fma(i64 %a, i64 %b, i64 %c) { ret i64 %a }
define internal i32 @__air_f64_lt(i64 %a, i64 %b) { ret i32 0 }
define internal i32 @__air_f64_to_f32_r(i64 %a, i32 %r) { ret i32 0 }
define internal i64 @__air_f64_sub_r(i64 %a, i64 %b, i32 %r) { ret i64 %a }
define internal i64 @__air_f64_div_r(i64 %a, i64 %b, i32 %r) { ret i64 %a }
