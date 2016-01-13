; RUN: llc -mtriple=armv8 -mcpu=cyclone < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOTSWIFT
; RUN: llc -mtriple=armv8 -mcpu=swift < %s | FileCheck %s --check-prefix=CHECK
; RUN: llc -mtriple=armv8 -mcpu=cortex-a57 < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOTSWIFT

declare arm_aapcs_vfpcc void @take_vec64(<2 x i32>)

define void @test_vec64() {
; CHECK-LABEL: test_vec64:

  call arm_aapcs_vfpcc void @take_vec64(<2 x i32> <i32 0, i32 0>)
  call arm_aapcs_vfpcc void @take_vec64(<2 x i32> <i32 0, i32 0>)
; CHECK-NOTSWIFT-NOT: vmov.f64 d0,
; CHECK: vmov.i32 d0, #0
; CHECK: bl
; CHECK: vmov.i32 d0, #0
; CHECK: bl

  ret void
}

declare arm_aapcs_vfpcc void @take_vec128(<8 x i16>)

define void @test_vec128() {
; CHECK-LABEL: test_vec128:

  call arm_aapcs_vfpcc void @take_vec128(<8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>)
  call arm_aapcs_vfpcc void @take_vec128(<8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>)
; CHECK-NOT: vmov.f64 [[ZEROREG:d[0-9]+]],
; CHECK: vmov.i32 q0, #0
; CHECK: bl
; CHECK: vmov.i32 q0, #0
; CHECK: bl

  ret void
}

declare void @take_i32(i32)

define void @test_i32() {
; CHECK-LABEL: test_i32:

  call arm_aapcs_vfpcc void @take_i32(i32 0)
  call arm_aapcs_vfpcc void @take_i32(i32 0)
; CHECK-NOTSWIFT-NOT: vmov.f64 [[ZEROREG:d[0-9]+]],
; CHECK: mov r0, #0
; CHECK: bl
; CHECK: mov r0, #0
; CHECK: bl

; It doesn't particularly matter what Swift does here, there isn't carefully
; crafted behaviour that we might break in Cyclone.

  ret void
}
