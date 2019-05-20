; RUN: llc < %s -mtriple=armv7-linux-gnueabihf -o - | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7em-none-eabi -mcpu=cortex-m4 | FileCheck %s --check-prefix=CHECK-M4F

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define arm_aapcs_vfpcc void @test_1float({ float } %a) {
  call arm_aapcs_vfpcc void @test_1float({ float } { float 1.0 })
  ret void

; CHECK-LABEL: test_1float:
; CHECK-DAG: vmov.f32 s0, #1.{{0+}}e+00
; CHECK: bl test_1float

; CHECK-M4F-LABEL: test_1float:
; CHECK-M4F-DAG: vmov.f32 s0, #1.{{0+}}e+00
; CHECK-M4F: bl test_1float
}

define arm_aapcs_vfpcc void @test_2float({ float, float } %a) {
  call arm_aapcs_vfpcc void @test_2float({ float, float } { float 1.0, float 2.0 })
  ret void

; CHECK-LABEL: test_2float:
; CHECK-DAG: vmov.f32 s0, #1.{{0+}}e+00
; CHECK-DAG: vmov.f32 s1, #2.{{0+}}e+00
; CHECK: bl test_2float

; CHECK-M4F-LABEL: test_2float:
; CHECK-M4F-DAG: vmov.f32 s0, #1.{{0+}}e+00
; CHECK-M4F-DAG: vmov.f32 s1, #2.{{0+}}e+00
; CHECK-M4F: bl test_2float
}

define arm_aapcs_vfpcc void @test_3float({ float, float, float } %a) {
  call arm_aapcs_vfpcc void @test_3float({ float, float, float } { float 1.0, float 2.0, float 3.0 })
  ret void

; CHECK-LABEL: test_3float:
; CHECK-DAG: vmov.f32 s0, #1.{{0+}}e+00
; CHECK-DAG: vmov.f32 s1, #2.{{0+}}e+00
; CHECK-DAG: vmov.f32 s2, #3.{{0+}}e+00
; CHECK: bl test_3float

; CHECK-M4F-LABEL: test_3float:
; CHECK-M4F-DAG: vmov.f32 s0, #1.{{0+}}e+00
; CHECK-M4F-DAG: vmov.f32 s1, #2.{{0+}}e+00
; CHECK-M4F-DAG: vmov.f32 s2, #3.{{0+}}e+00
; CHECK-M4F: bl test_3float
}

define arm_aapcs_vfpcc void @test_1double({ double } %a) {
; CHECK-LABEL: test_1double:
; CHECK-DAG: vmov.f64 d0, #1.{{0+}}e+00
; CHECK: bl test_1double

; CHECK-M4F-LABEL: test_1double:
; CHECK-M4F: vldr d0, [[CP_LABEL:.*]]
; CHECK-M4F: bl test_1double
; CHECK-M4F: [[CP_LABEL]]
; CHECK-M4F-NEXT: .long 0
; CHECK-M4F-NEXT: .long 1072693248

  call arm_aapcs_vfpcc void @test_1double({ double } { double 1.0 })
  ret void
}

; Final double argument might be put in s15 & [sp] if we're careless. It should
; go all on the stack.
define arm_aapcs_vfpcc void @test_1double_nosplit([4 x float], [4 x double], [3 x float], double %a) {
; CHECK-LABEL: test_1double_nosplit:
; CHECK-DAG: mov [[ONELO:r[0-9]+]], #0
; CHECK-DAG: movw [[ONEHI:r[0-9]+]], #0
; CHECK-DAG: movt [[ONEHI]], #16368
; CHECK: strd [[ONELO]], [[ONEHI]], [sp]
; CHECK: bl test_1double_nosplit

; CHECK-M4F-LABEL: test_1double_nosplit:
; CHECK-M4F: movs [[ONEHI:r[0-9]+]], #0
; CHECK-M4F: movs [[ONELO:r[0-9]+]], #0
; CHECK-M4F: movt [[ONEHI]], #16368
; CHECK-M4F: strd [[ONELO]], [[ONEHI]], [sp]
; CHECK-M4F: bl test_1double_nosplit
  call arm_aapcs_vfpcc void @test_1double_nosplit([4 x float] undef, [4 x double] undef, [3 x float] undef, double 1.0)
  ret void
}

; Final double argument might go at [sp, #4] if we're careless. Should go at
; [sp, #8] to preserve alignment.
define arm_aapcs_vfpcc void @test_1double_misaligned([4 x double], [4 x double], float, double) {
  call arm_aapcs_vfpcc void @test_1double_misaligned([4 x double] undef, [4 x double] undef, float undef, double 1.0)

; CHECK-LABEL: test_1double_misaligned:
; CHECK-DAG: movw [[ONEHI:r[0-9]+]], #0
; CHECK-DAG: mov [[ONELO:r[0-9]+]], #0
; CHECK-DAG: movt [[ONEHI]], #16368
; CHECK-DAG: strd [[ONELO]], [[ONEHI]], [sp, #8]

; CHECK-M4F-LABEL: test_1double_misaligned:
; CHECK-M4F: movs [[ONEHI:r[0-9]+]], #0
; CHECK-M4F: movs [[ONELO:r[0-9]+]], #0
; CHECK-M4F: movt [[ONEHI]], #16368
; CHECK-M4F: strd [[ONELO]], [[ONEHI]], [sp, #8]
; CHECK-M4F: bl test_1double_misaligned

  ret void
}
