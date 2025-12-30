; RUN: llc < %s -mtriple=thumbv7s-apple-ios7.0.0 | FileCheck %s

define arm_aapcs_vfpcc { <8 x half>, <8 x half> } @f1() {
; CHECK-LABEL: _f1
; CHECK:      vpush   {d8, d9, d10, d11}
; CHECK-NEXT: vmov.i32        q8, #0x0
; CHECK-NEXT: vmov.u16        r0, d16[0]
; CHECK-NEXT: vmov    d4, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[1]
; CHECK-NEXT: vmov    d8, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[2]
; CHECK-NEXT: vmov    d5, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[3]
; CHECK-NEXT: vmov    d9, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[0]
; CHECK-NEXT: vmov    d6, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[1]
; CHECK-NEXT: vmov    d10, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[2]
; CHECK-NEXT: vmov    d7, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[3]
; CHECK-NEXT: vmov    d11, r0, r0
; CHECK:      vmov.f32        s0, s8
; CHECK:      vmov.f32        s1, s16
; CHECK:      vmov.f32        s2, s10
; CHECK:      vmov.f32        s3, s18
; CHECK:      vmov.f32        s4, s12
; CHECK:      vmov.f32        s5, s20
; CHECK:      vmov.f32        s6, s14
; CHECK:      vmov.f32        s7, s22
; CHECK:      vmov.f32        s9, s16
; CHECK:      vmov.f32        s11, s18
; CHECK:      vmov.f32        s13, s20
; CHECK:      vmov.f32        s15, s22
; CHECK:      vpop    {d8, d9, d10, d11}
; CHECK-NEXT: bx      lr

  ret { <8 x half>, <8 x half> } zeroinitializer
}

define swiftcc { <8 x half>, <8 x half> } @f2() {
; CHECK-LABEL: _f2
; CHECK:      vpush   {d8, d9, d10, d11}
; CHECK-NEXT: vmov.i32        q8, #0x0
; CHECK-NEXT: vmov.u16        r0, d16[0]
; CHECK-NEXT: vmov    d4, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[1]
; CHECK-NEXT: vmov    d8, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[2]
; CHECK-NEXT: vmov    d5, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[3]
; CHECK-NEXT: vmov    d9, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[0]
; CHECK-NEXT: vmov    d6, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[1]
; CHECK-NEXT: vmov    d10, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[2]
; CHECK-NEXT: vmov    d7, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[3]
; CHECK-NEXT: vmov    d11, r0, r0
; CHECK:      vmov.f32        s0, s8
; CHECK:      vmov.f32        s1, s16
; CHECK:      vmov.f32        s2, s10
; CHECK:      vmov.f32        s3, s18
; CHECK:      vmov.f32        s4, s12
; CHECK:      vmov.f32        s5, s20
; CHECK:      vmov.f32        s6, s14
; CHECK:      vmov.f32        s7, s22
; CHECK:      vmov.f32        s9, s16
; CHECK:      vmov.f32        s11, s18
; CHECK:      vmov.f32        s13, s20
; CHECK:      vmov.f32        s15, s22
; CHECK-NEXT: vpop    {d8, d9, d10, d11}
; CHECK-NEXT: bx      lr

  ret { <8 x half>, <8 x half> } zeroinitializer
}

define fastcc { <8 x half>, <8 x half> } @f3() {
; CHECK-LABEL: _f3
; CHECK:      vpush   {d8, d9, d10, d11}
; CHECK-NEXT: vmov.i32        q8, #0x0
; CHECK-NEXT: vmov.u16        r0, d16[0]
; CHECK-NEXT: vmov    d4, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[1]
; CHECK-NEXT: vmov    d8, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[2]
; CHECK-NEXT: vmov    d5, r0, r0
; CHECK-NEXT: vmov.u16        r0, d16[3]
; CHECK-NEXT: vmov    d9, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[0]
; CHECK-NEXT: vmov    d6, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[1]
; CHECK-NEXT: vmov    d10, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[2]
; CHECK-NEXT: vmov    d7, r0, r0
; CHECK-NEXT: vmov.u16        r0, d17[3]
; CHECK-NEXT: vmov    d11, r0, r0
; CHECK:      vmov.f32        s0, s8
; CHECK:      vmov.f32        s1, s16
; CHECK:      vmov.f32        s2, s10
; CHECK:      vmov.f32        s3, s18
; CHECK:      vmov.f32        s4, s12
; CHECK:      vmov.f32        s5, s20
; CHECK:      vmov.f32        s6, s14
; CHECK:      vmov.f32        s7, s22
; CHECK:      vmov.f32        s9, s16
; CHECK:      vmov.f32        s11, s18
; CHECK:      vmov.f32        s13, s20
; CHECK:      vmov.f32        s15, s22
; CHECK-NEXT: vpop    {d8, d9, d10, d11}
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
