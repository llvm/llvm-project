; RUN: llc -mcpu=swift -mtriple=thumbv7s-apple-ios -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-STRIDE4
; RUN: llc -mcpu=swift -mtriple=thumbv7k-apple-watchos -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-STRIDE4-WATCH
; RUN: llc -mcpu=cortex-a57 -mtriple=thumbv7-linux-gnueabihf -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-GENERIC
; RUN: llc -mattr=wide-stride-vfp -mtriple=thumbv7-linux-gnueabihf -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-GENERIC4

; CHECK-LABEL: test_reg_stride:
define void @test_reg_stride(double %a, double %b) {
; CHECK-STRIDE4-DAG: vmov d16, r
; CHECK-STRIDE4-DAG: vmov d18, r

; CHECK-STRIDE4-WATCH-DAG: vmov.f64 d16, d
; CHECK-STRIDE4-WATCH-DAG: vmov.f64 d18, d

; CHECK-GENERIC-DAG: vmov.f64 d16, {{d[01]}}
; CHECK-GENERIC-DAG: vmov.f64 d17, {{d[01]}}

; CHECK-GENERIC4-DAG: vmov.f64 d16, {{d[01]}}
; CHECK-GENERIC4-DAG: vmov.f64 d18, {{d[01]}}

  call void asm "", "~{r0},~{r1},~{d0},~{d1}"()
  call arm_aapcs_vfpcc void @eat_doubles(double %a, double %b)
  ret void
}

; CHECK-LABEL: test_stride_minsize:
define void @test_stride_minsize(float %a, float %b) minsize {
; CHECK-STRIDE4: vmov d2, {{r[01]}}
; CHECK-STRIDE4: vmov d3, {{r[01]}}

; CHECK-STRIDE4-WATCH-DAG: vmov.f32 s4, {{s[01]}}
; CHECK-STRIDE4-WATCH-DAG: vmov.f32 s8, {{s[01]}}

; CHECK-GENERIC-DAG: vmov.f32 s4, {{s[01]}}
; CHECK-GENERIC-DAG: vmov.f32 s6, {{s[01]}}

; CHECK-GENERIC4-DAG: vmov.f32 s4, {{s[01]}}
; CHECK-GENERIC4-DAG: vmov.f32 s6, {{s[01]}}

  call void asm "", "~{r0},~{r1},~{s0},~{s1},~{d0},~{d1}"()
  call arm_aapcs_vfpcc void @eat_floats(float %a, float %b)
  ret void
}

declare arm_aapcs_vfpcc void @eat_doubles(double, double)
declare arm_aapcs_vfpcc void @eat_floats(float, float)
