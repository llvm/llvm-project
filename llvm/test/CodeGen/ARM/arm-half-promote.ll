; RUN: llc < %s -mtriple=thumbv7s-apple-ios7.0.0 | FileCheck %s

define arm_aapcs_vfpcc { <8 x half>, <8 x half> } @f1() {
; CHECK-LABEL: _f1
; CHECK: vpush   {d8}
; CHECK-NEXT: vmov.f64        d8, #5.000000e-01
; CHECK-NEXT: vmov.i32        d8, #0x0
; CHECK-NEXT: vmov.i32        d0, #0x0
; CHECK-NEXT: vmov.i32        d1, #0x0
; CHECK-NEXT: vmov.i32        d2, #0x0
; CHECK-NEXT: vmov.i32        d3, #0x0
; CHECK-NEXT: vmov.i32        d4, #0x0
; CHECK-NEXT: vmov.i32        d5, #0x0
; CHECK-NEXT: vmov.i32        d6, #0x0
; CHECK-NEXT: vmov.i32        d7, #0x0
; CHECK-NEXT: vmov.f32        s1, s16
; CHECK-NEXT: vmov.f32        s3, s16
; CHECK-NEXT: vmov.f32        s5, s16
; CHECK-NEXT: vmov.f32        s7, s16
; CHECK-NEXT: vmov.f32        s9, s16
; CHECK-NEXT: vmov.f32        s11, s16
; CHECK-NEXT: vmov.f32        s13, s16
; CHECK-NEXT: vmov.f32        s15, s16
; CHECK-NEXT: vpop    {d8}
; CHECK-NEXT: bx      lr
  ret { <8 x half>, <8 x half> } zeroinitializer
}

define swiftcc { <8 x half>, <8 x half> } @f2() {
; CHECK-LABEL: _f2
; CHECK: vpush   {d8}
; CHECK-NEXT: vmov.f64        d8, #5.000000e-01
; CHECK-NEXT: vmov.i32        d8, #0x0
; CHECK-NEXT: vmov.i32        d0, #0x0
; CHECK-NEXT: vmov.i32        d1, #0x0
; CHECK-NEXT: vmov.i32        d2, #0x0
; CHECK-NEXT: vmov.i32        d3, #0x0
; CHECK-NEXT: vmov.i32        d4, #0x0
; CHECK-NEXT: vmov.i32        d5, #0x0
; CHECK-NEXT: vmov.i32        d6, #0x0
; CHECK-NEXT: vmov.i32        d7, #0x0
; CHECK-NEXT: vmov.f32        s1, s16
; CHECK-NEXT: vmov.f32        s3, s16
; CHECK-NEXT: vmov.f32        s5, s16
; CHECK-NEXT: vmov.f32        s7, s16
; CHECK-NEXT: vmov.f32        s9, s16
; CHECK-NEXT: vmov.f32        s11, s16
; CHECK-NEXT: vmov.f32        s13, s16
; CHECK-NEXT: vmov.f32        s15, s16
; CHECK-NEXT: vpop    {d8}
; CHECK-NEXT: bx      lr
  ret { <8 x half>, <8 x half> } zeroinitializer
}

define fastcc { <8 x half>, <8 x half> } @f3() {
; CHECK-LABEL: _f3
; CHECK: vpush   {d8}
; CHECK-NEXT: vmov.f64        d8, #5.000000e-01
; CHECK-NEXT: vmov.i32        d8, #0x0
; CHECK-NEXT: vmov.i32        d0, #0x0
; CHECK-NEXT: vmov.i32        d1, #0x0
; CHECK-NEXT: vmov.i32        d2, #0x0
; CHECK-NEXT: vmov.i32        d3, #0x0
; CHECK-NEXT: vmov.i32        d4, #0x0
; CHECK-NEXT: vmov.i32        d5, #0x0
; CHECK-NEXT: vmov.i32        d6, #0x0
; CHECK-NEXT: vmov.i32        d7, #0x0
; CHECK-NEXT: vmov.f32        s1, s16
; CHECK-NEXT: vmov.f32        s3, s16
; CHECK-NEXT: vmov.f32        s5, s16
; CHECK-NEXT: vmov.f32        s7, s16
; CHECK-NEXT: vmov.f32        s9, s16
; CHECK-NEXT: vmov.f32        s11, s16
; CHECK-NEXT: vmov.f32        s13, s16
; CHECK-NEXT: vmov.f32        s15, s16
; CHECK-NEXT: vpop    {d8}
; CHECK-NEXT: bx      lr

  ret { <8 x half>, <8 x half> } zeroinitializer
}

define void @extract_insert(ptr %dst) optnone noinline {
; CHECK-LABEL: extract_insert:
; CHECK: vmov.i32 d0, #0x0
; CHECK: vcvtb.f16.f32 s0, s0
; CHECK: vmov r1, s0
; CHECK: strh r1, [r0]
  %splat.splatinsert = insertelement <1 x half> zeroinitializer, half 0xH0000, i32 0
  br label %next

next:
  store <1 x half> %splat.splatinsert, ptr %dst
  ret void
}
