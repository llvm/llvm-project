; RUN: llc -mtriple=thumbv8.1m.main -mattr=+cdecp0 -mattr=+cdecp1 -mattr=+mve -verify-machineinstrs -o - %s | FileCheck %s
; RUN: llc -mtriple=thumbv8m.main -mattr=+cdecp0 -mattr=+cdecp1 -mattr=+fp-armv8d16sp -verify-machineinstrs -o - %s | FileCheck %s

declare float @llvm.arm.cde.vcx1.f32(i32 immarg, i32 immarg)
declare float @llvm.arm.cde.vcx1a.f32(i32 immarg, float, i32 immarg)
declare float @llvm.arm.cde.vcx2.f32(i32 immarg, float, i32 immarg)
declare float @llvm.arm.cde.vcx2a.f32(i32 immarg, float, float, i32 immarg)
declare float @llvm.arm.cde.vcx3.f32(i32 immarg, float, float, i32 immarg)
declare float @llvm.arm.cde.vcx3a.f32(i32 immarg, float, float, float, i32 immarg)

declare double @llvm.arm.cde.vcx1.f64(i32 immarg, i32 immarg)
declare double @llvm.arm.cde.vcx1a.f64(i32 immarg, double, i32 immarg)
declare double @llvm.arm.cde.vcx2.f64(i32 immarg, double, i32 immarg)
declare double @llvm.arm.cde.vcx2a.f64(i32 immarg, double, double, i32 immarg)
declare double @llvm.arm.cde.vcx3.f64(i32 immarg, double, double, i32 immarg)
declare double @llvm.arm.cde.vcx3a.f64(i32 immarg, double, double, double, i32 immarg)

define arm_aapcs_vfpcc i32 @test_vcx1_u32() {
; CHECK-LABEL: test_vcx1_u32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vcx1 p0, s0, #11
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
entry:
  %0 = call float @llvm.arm.cde.vcx1.f32(i32 0, i32 11)
  %1 = bitcast float %0 to i32
  ret i32 %1
}

define arm_aapcs_vfpcc i32 @test_vcx1a_u32(i32 %acc) {
; CHECK-LABEL: test_vcx1a_u32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov s0, r0
; CHECK-NEXT:    vcx1a p1, s0, #12
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i32 %acc to float
  %1 = call float @llvm.arm.cde.vcx1a.f32(i32 1, float %0, i32 12)
  %2 = bitcast float %1 to i32
  ret i32 %2
}

define arm_aapcs_vfpcc i32 @test_vcx2_u32(i32 %n) {
; CHECK-LABEL: test_vcx2_u32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov s0, r0
; CHECK-NEXT:    vcx2 p0, s0, s0, #21
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i32 %n to float
  %1 = call float @llvm.arm.cde.vcx2.f32(i32 0, float %0, i32 21)
  %2 = bitcast float %1 to i32
  ret i32 %2
}

define arm_aapcs_vfpcc i32 @test_vcx2a_u32(i32 %acc, i32 %n) {
; CHECK-LABEL: test_vcx2a_u32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov s0, r1
; CHECK-NEXT:    vmov s2, r0
; CHECK-NEXT:    vcx2a p0, s2, s0, #22
; CHECK-NEXT:    vmov r0, s2
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i32 %acc to float
  %1 = bitcast i32 %n to float
  %2 = call float @llvm.arm.cde.vcx2a.f32(i32 0, float %0, float %1, i32 22)
  %3 = bitcast float %2 to i32
  ret i32 %3
}

define arm_aapcs_vfpcc i32 @test_vcx3_u32(i32 %n, i32 %m) {
; CHECK-LABEL: test_vcx3_u32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov s0, r1
; CHECK-NEXT:    vmov s2, r0
; CHECK-NEXT:    vcx3 p1, s0, s2, s0, #3
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i32 %n to float
  %1 = bitcast i32 %m to float
  %2 = call float @llvm.arm.cde.vcx3.f32(i32 1, float %0, float %1, i32 3)
  %3 = bitcast float %2 to i32
  ret i32 %3
}

define arm_aapcs_vfpcc i32 @test_vcx3a_u32(i32 %acc, i32 %n, i32 %m) {
; CHECK-LABEL: test_vcx3a_u32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov s0, r2
; CHECK-NEXT:    vmov s2, r1
; CHECK-NEXT:    vmov s4, r0
; CHECK-NEXT:    vcx3a p0, s4, s2, s0, #5
; CHECK-NEXT:    vmov r0, s4
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i32 %acc to float
  %1 = bitcast i32 %n to float
  %2 = bitcast i32 %m to float
  %3 = call float @llvm.arm.cde.vcx3a.f32(i32 0, float %0, float %1, float %2, i32 5)
  %4 = bitcast float %3 to i32
  ret i32 %4
}

define arm_aapcs_vfpcc i64 @test_vcx1d_u64() {
; CHECK-LABEL: test_vcx1d_u64:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vcx1 p0, d0, #11
; CHECK-NEXT:    vmov r0, r1, d0
; CHECK-NEXT:    bx lr
entry:
  %0 = call double @llvm.arm.cde.vcx1.f64(i32 0, i32 11)
  %1 = bitcast double %0 to i64
  ret i64 %1
}

define arm_aapcs_vfpcc i64 @test_vcx1da_u64(i64 %acc) {
; CHECK-LABEL: test_vcx1da_u64:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov d0, r0, r1
; CHECK-NEXT:    vcx1a p1, d0, #12
; CHECK-NEXT:    vmov r0, r1, d0
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i64 %acc to double
  %1 = call double @llvm.arm.cde.vcx1a.f64(i32 1, double %0, i32 12)
  %2 = bitcast double %1 to i64
  ret i64 %2
}

define arm_aapcs_vfpcc i64 @test_vcx2d_u64(i64 %n) {
; CHECK-LABEL: test_vcx2d_u64:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov d0, r0, r1
; CHECK-NEXT:    vcx2 p0, d0, d0, #21
; CHECK-NEXT:    vmov r0, r1, d0
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i64 %n to double
  %1 = call double @llvm.arm.cde.vcx2.f64(i32 0, double %0, i32 21)
  %2 = bitcast double %1 to i64
  ret i64 %2
}

define arm_aapcs_vfpcc i64 @test_vcx2da_u64(i64 %acc, i64 %n) {
; CHECK-LABEL: test_vcx2da_u64:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov d0, r2, r3
; CHECK-NEXT:    vmov d1, r0, r1
; CHECK-NEXT:    vcx2a p0, d1, d0, #22
; CHECK-NEXT:    vmov r0, r1, d1
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i64 %acc to double
  %1 = bitcast i64 %n to double
  %2 = call double @llvm.arm.cde.vcx2a.f64(i32 0, double %0, double %1, i32 22)
  %3 = bitcast double %2 to i64
  ret i64 %3
}

define arm_aapcs_vfpcc i64 @test_vcx3d_u64(i64 %n, i64 %m) {
; CHECK-LABEL: test_vcx3d_u64:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmov d0, r2, r3
; CHECK-NEXT:    vmov d1, r0, r1
; CHECK-NEXT:    vcx3 p1, d0, d1, d0, #3
; CHECK-NEXT:    vmov r0, r1, d0
; CHECK-NEXT:    bx lr
entry:
  %0 = bitcast i64 %n to double
  %1 = bitcast i64 %m to double
  %2 = call double @llvm.arm.cde.vcx3.f64(i32 1, double %0, double %1, i32 3)
  %3 = bitcast double %2 to i64
  ret i64 %3
}

define arm_aapcs_vfpcc i64 @test_vcx3da_u64(i64 %acc, i64 %n, i64 %m) {
; CHECK-LABEL: test_vcx3da_u64:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    push {r7, lr}
; CHECK-NEXT:    ldrd lr, r12, [sp, #8]
; CHECK-DAG:     vmov [[D0:d.*]], r0, r1
; CHECK-DAG:     vmov [[D1:d.*]], r2, r3
; CHECK-DAG:     vmov [[D2:d.*]], lr, r12
; CHECK-NEXT:    vcx3a p0, [[D0]], [[D1]], [[D2]], #5
; CHECK-NEXT:    vmov r0, r1, [[D0]]
; CHECK-NEXT:    pop {r7, pc}
entry:
  %0 = bitcast i64 %acc to double
  %1 = bitcast i64 %n to double
  %2 = bitcast i64 %m to double
  %3 = call double @llvm.arm.cde.vcx3a.f64(i32 0, double %0, double %1, double %2, i32 5)
  %4 = bitcast double %3 to i64
  ret i64 %4
}
