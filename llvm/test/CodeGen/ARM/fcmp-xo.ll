; RUN: llc -mtriple=thumbv7m-arm-none-eabi -mattr=+execute-only,+fp-armv8 %s -o - | FileCheck --check-prefixes=CHECK,VMOVSR %s
; RUN: llc -mtriple=thumbv7m-arm-none-eabi -mattr=+execute-only,+fp-armv8,+neon,+neonfp %s -o - | FileCheck --check-prefixes=CHECK,NEON %s

define arm_aapcs_vfpcc float @foo0() local_unnamed_addr {
  %1 = fcmp nsz olt float undef, 0.000000e+00
  %2 = select i1 %1, float -5.000000e-01, float 5.000000e-01
  ret float %2
}
; CHECK-LABEL: foo0
; CHECK: vcmpe.f32 {{s[0-9]+}}, #0


define arm_aapcs_vfpcc float @float1() local_unnamed_addr {
  br i1 undef, label %.end, label %1

  %2 = fcmp nsz olt float undef, 1.000000e+00
  %3 = select i1 %2, float -5.000000e-01, float 5.000000e-01
  br label %.end

.end:
  %4 = phi float [ undef, %0 ], [ %3, %1]
  ret float %4
}
; CHECK-LABEL: float1
; CHECK: vmov.f32 [[FPREG:s[0-9]+]], #1.000000e+00
; CHECK: vcmpe.f32 [[FPREG]], {{s[0-9]+}}

define arm_aapcs_vfpcc float @float128() local_unnamed_addr {
  %1 = fcmp nsz olt float undef, 128.000000e+00
  %2 = select i1 %1, float -5.000000e-01, float 5.000000e-01
  ret float %2
}
; CHECK-LABEL: float128
; CHECK: mov.w [[REG:r[0-9]+]], #1124073472
; VMOVSR: vmov [[FPREG:s[0-9]+]], [[REG]]
; VMOVSR: vcmpe.f32 [[FPREG]], {{s[0-9]+}}
; NEON: vmov d2, [[REG]], [[REG]]
; NEON: vcmpe.f32 s4, {{s[0-9]+}}


define arm_aapcs_vfpcc double @double1() local_unnamed_addr {
  %1 = fcmp nsz olt double undef, 1.000000e+00
  %2 = select i1 %1, double -5.000000e-01, double 5.000000e-01
  ret double %2
}
; CHECK-LABEL: double1
; CHECK: vmov.f64 [[FPREG:d[0-9]+]], #1.000000e+00
; CHECK: vcmpe.f64 [[FPREG]], {{d[0-9]+}}

define arm_aapcs_vfpcc double @double128() local_unnamed_addr {
  %1 = fcmp nsz olt double undef, 128.000000e+00
  %2 = select i1 %1, double -5.000000e-01, double 5.000000e-01
  ret double %2
}
; CHECK-LABEL: double128
; CHECK: movs [[REGH:r[0-9]+]], #0
; CHECK: movt [[REGH]], #16480
; CHECK: movs [[REGL:r[0-9]+]], #0
; CHECK: vmov [[FPREG:d[0-9]+]], [[REGL]], [[REGH]]
; CHECK: vcmpe.f64 [[FPREG]], {{d[0-9]+}}

