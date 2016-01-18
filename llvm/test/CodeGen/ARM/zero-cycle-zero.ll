; RUN: llc -mtriple=armv8 -mcpu=cyclone < %s | FileCheck %s --check-prefix=CHECK-CYCLONE
; RUN: llc -mtriple=armv8 -mcpu=swift < %s | FileCheck %s --check-prefix=CHECK-SWIFT

declare arm_aapcs_vfpcc void @take_vec64(<2 x i32>)

define void @test_vec64() {
; CHECK-CYCLONE-LABEL: test_vec64:
; CHECK-SWIFT-LABEL: test_vec64:

  call arm_aapcs_vfpcc void @take_vec64(<2 x i32> <i32 0, i32 0>)
  call arm_aapcs_vfpcc void @take_vec64(<2 x i32> <i32 0, i32 0>)
; CHECK-CYCLONE-NOT: vmov.f64 d0,
; CHECK-CYCLONE: vmov.i32 d0, #0
; CHECK-CYCLONE: bl
; CHECK-CYCLONE: vmov.i32 d0, #0
; CHECK-CYCLONE: bl

; CHECK-SWIFT: vmov.f64 [[ZEROREG:d[0-9]+]],
; CHECK-SWIFT: vmov.i32 [[ZEROREG]], #0
; CHECK-SWIFT: vorr d0, [[ZEROREG]], [[ZEROREG]]
; CHECK-SWIFT: bl
; CHECK-SWIFT: vorr d0, [[ZEROREG]], [[ZEROREG]]
; CHECK-SWIFT: bl

  ret void
}

declare arm_aapcs_vfpcc void @take_vec128(<8 x i16>)

define void @test_vec128() {
; CHECK-CYCLONE-LABEL: test_vec128:
; CHECK-SWIFT-LABEL: test_vec128:

  call arm_aapcs_vfpcc void @take_vec128(<8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>)
  call arm_aapcs_vfpcc void @take_vec128(<8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>)
; CHECK-CYCLONE-NOT: vmov.f64 [[ZEROREG:d[0-9]+]],
; CHECK-CYCLONE: vmov.i32 q0, #0
; CHECK-CYCLONE: bl
; CHECK-CYCLONE: vmov.i32 q0, #0
; CHECK-CYCLONE: bl

; CHECK-SWIFT-NOT: vmov.f64 [[ZEROREG:d[0-9]+]],
; CHECK-SWIFT: vmov.i32 [[ZEROREG:q[0-9]+]], #0
; CHECK-SWIFT: vorr q0, [[ZEROREG]], [[ZEROREG]]
; CHECK-SWIFT: bl
; CHECK-SWIFT: vorr q0, [[ZEROREG]], [[ZEROREG]]
; CHECK-SWIFT: bl

  ret void
}

declare void @take_i32(i32)

define void @test_i32() {
; CHECK-CYCLONE-LABEL: test_i32:
; CHECK-SWIFT-LABEL: test_i32:

  call arm_aapcs_vfpcc void @take_i32(i32 0)
  call arm_aapcs_vfpcc void @take_i32(i32 0)
; CHECK-CYCLONE-NOT: vmov.f64 [[ZEROREG:d[0-9]+]],
; CHECK-CYCLONE: mov r0, #0
; CHECK-CYCLONE: bl
; CHECK-CYCLONE: mov r0, #0
; CHECK-CYCLONE: bl

; It doesn't particularly matter what Swift does here, there isn't carefully
; crafted behaviour that we might break in Cyclone.

  ret void
}
