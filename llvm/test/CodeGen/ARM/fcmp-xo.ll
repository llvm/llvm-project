; RUN: llc -mtriple=thumbv7m-arm-none-eabi -mattr=+execute-only,+fp-armv8 %s -o - | FileCheck %s

; This function used to run into a code selection error on fp-armv8 due to
; different ordering of the constant arguments of fcmp. Fixed by extending the
; code selection to handle the missing case.
define arm_aapcs_vfpcc void @foo0() local_unnamed_addr {
  br i1 undef, label %.end, label %1

  %2 = fcmp nsz olt float undef, 0.000000e+00
  %3 = select i1 %2, float -5.000000e-01, float 5.000000e-01
  %4 = fadd nsz float undef, %3
  %5 = fptosi float %4 to i32
  %6 = ashr i32 %5, 4
  %7 = icmp slt i32 %6, 0
  br i1 %7, label %8, label %.end

  tail call arm_aapcs_vfpcc void @bar()
  br label %.end

.end:
  ret void
}
; CHECK-LABEL: foo0
; CHECK: vcmpe.f32 {{s[0-9]+}}, #0


define arm_aapcs_vfpcc void @float1() local_unnamed_addr {
  br i1 undef, label %.end, label %1

  %2 = fcmp nsz olt float undef, 1.000000e+00
  %3 = select i1 %2, float -5.000000e-01, float 5.000000e-01
  %4 = fadd nsz float undef, %3
  %5 = fptosi float %4 to i32
  %6 = ashr i32 %5, 4
  %7 = icmp slt i32 %6, 0
  br i1 %7, label %8, label %.end

  tail call arm_aapcs_vfpcc void @bar()
  br label %.end

.end:
  ret void
}
; CHECK-LABEL: float1
; CHECK: vmov.f32 [[FPREG:s[0-9]+]], #1.000000e+00
; CHECK: vcmpe.f32 [[FPREG]], {{s[0-9]+}}

define arm_aapcs_vfpcc void @float128() local_unnamed_addr {
  br i1 undef, label %.end, label %1

  %2 = fcmp nsz olt float undef, 128.000000e+00
  %3 = select i1 %2, float -5.000000e-01, float 5.000000e-01
  %4 = fadd nsz float undef, %3
  %5 = fptosi float %4 to i32
  %6 = ashr i32 %5, 4
  %7 = icmp slt i32 %6, 0
  br i1 %7, label %8, label %.end

  tail call arm_aapcs_vfpcc void @bar()
  br label %.end

.end:
  ret void
}
; CHECK-LABEL: float128
; CHECK: mov.w [[REG:r[0-9]+]], #1124073472
; CHECK: vmov [[FPREG:s[0-9]+]], [[REG]]
; CHECK: vcmpe.f32 [[FPREG]], {{s[0-9]+}}


define arm_aapcs_vfpcc void @double1() local_unnamed_addr {
  br i1 undef, label %.end, label %1

  %2 = fcmp nsz olt double undef, 1.000000e+00
  %3 = select i1 %2, double -5.000000e-01, double 5.000000e-01
  %4 = fadd nsz double undef, %3
  %5 = fptosi double %4 to i32
  %6 = ashr i32 %5, 4
  %7 = icmp slt i32 %6, 0
  br i1 %7, label %8, label %.end

  tail call arm_aapcs_vfpcc void @bar()
  br label %.end

.end:
  ret void
}
; CHECK-LABEL: double1
; CHECK: vmov.f64 [[FPREG:d[0-9]+]], #1.000000e+00
; CHECK: vcmpe.f64 [[FPREG]], {{d[0-9]+}}

define arm_aapcs_vfpcc void @double128() local_unnamed_addr {
  br i1 undef, label %.end, label %1

  %2 = fcmp nsz olt double undef, 128.000000e+00
  %3 = select i1 %2, double -5.000000e-01, double 5.000000e-01
  %4 = fadd nsz double undef, %3
  %5 = fptosi double %4 to i32
  %6 = ashr i32 %5, 4
  %7 = icmp slt i32 %6, 0
  br i1 %7, label %8, label %.end

  tail call arm_aapcs_vfpcc void @bar()
  br label %.end

.end:
  ret void
}
; CHECK-LABEL: double128
; CHECK: movs [[REGL:r[0-9]+]], #0
; CHECK: movs [[REGH:r[0-9]+]], #0
; CHECK: movt [[REGH]], #16480
; CHECK: vmov [[FPREG:d[0-9]+]], [[REGL]], [[REGH]]
; CHECK: vcmpe.f64 [[FPREG]], {{d[0-9]+}}


declare arm_aapcs_vfpcc void @bar() local_unnamed_addr

