; SOFT:
; RUN: llc < %s -mtriple=arm-none-eabi -float-abi=soft     | FileCheck %s --check-prefixes=CHECK,CHECK-SOFT

; SOFTFP:
; RUN: llc < %s -mtriple=arm-none-eabi -mattr=+vfp3        | FileCheck %s --check-prefixes=CHECK,CHECK-SOFTFP-VFP3
; RUN: llc < %s -mtriple=arm-none-eabi -mattr=+vfp4        | FileCheck %s --check-prefixes=CHECK,CHECK-SOFTFP-FP16
; RUN: llc < %s -mtriple=arm-none-eabi -mattr=+fullfp16    | FileCheck %s --check-prefixes=CHECK,CHECK-SOFTFP-FULLFP16

; HARD:
; RUN: llc < %s -mtriple=arm-none-eabihf -mattr=+vfp3      | FileCheck %s --check-prefixes=CHECK,CHECK-HARDFP-VFP3
; RUN: llc < %s -mtriple=arm-none-eabihf -mattr=+vfp4      | FileCheck %s --check-prefixes=CHECK,CHECK-HARDFP-FP16
; RUN: llc < %s -mtriple=arm-none-eabihf -mattr=+fullfp16  | FileCheck %s --check-prefixes=CHECK,CHECK-HARDFP-FULLFP16


define float @RetValBug(float %A.coerce) local_unnamed_addr {
entry:
  ret float undef
; This expression is optimised away due to the undef value. Check that
; LowerReturn can handle undef nodes (i.e. nodes which do not have any
; operands) when FullFP16 is enabled.
;
; CHECK-LABEL:            RetValBug:
; CHECK-HARDFP-FULLFP16:  mov pc, lr
}

define float @Add(float %a.coerce, float %b.coerce) local_unnamed_addr {
entry:
  %0 = bitcast float %a.coerce to i32
  %tmp.0.extract.trunc = trunc i32 %0 to i16
  %1 = bitcast i16 %tmp.0.extract.trunc to half
  %2 = bitcast float %b.coerce to i32
  %tmp1.0.extract.trunc = trunc i32 %2 to i16
  %3 = bitcast i16 %tmp1.0.extract.trunc to half
  %add = fadd half %1, %3
  %4 = bitcast half %add to i16
  %tmp4.0.insert.ext = zext i16 %4 to i32
  %5 = bitcast i32 %tmp4.0.insert.ext to float
  ret float %5

; CHECK-LABEL: Add:

; CHECK-SOFT:  bl  __aeabi_h2f
; CHECK-SOFT:  bl  __aeabi_h2f
; CHECK-SOFT:  bl  __aeabi_fadd
; CHECK-SOFT:  bl  __aeabi_f2h

; CHECK-SOFTFP-VFP3:  bl  __aeabi_h2f
; CHECK-SOFTFP-VFP3:  bl  __aeabi_h2f
; CHECK-SOFTFP-VFP3:  vadd.f32
; CHECK-SOFTFP-VFP3:  bl  __aeabi_f2h

; CHECK-SOFTFP-FP16:  vmov          [[S2:s[0-9]]], r1
; CHECK-SOFTFP-FP16:  vmov          [[S0:s[0-9]]], r0
; CHECK-SOFTFP-FP16:  vcvtb.f32.f16 [[S2]], [[S2]]
; CHECK-SOFTFP-FP16:  vcvtb.f32.f16 [[S0]], [[S0]]
; CHECK-SOFTFP-FP16:  vadd.f32      [[S0]], [[S0]], [[S2]]
; CHECK-SOFTFP-FP16:  vcvtb.f16.f32 [[S0]], [[S0]]
; CHECK-SOFTFP-FP16:  vmov  r0, s0

; CHECK-SOFTFP-FULLFP16:       vmov.f16  [[S0:s[0-9]]], r1
; CHECK-SOFTFP-FULLFP16:       vmov.f16  [[S2:s[0-9]]], r0
; CHECK-SOFTFP-FULLFP16:       vadd.f16  [[S0]], [[S2]], [[S0]]
; CHECK-SOFTFP-FULLFP16-NEXT:  vmov.f16  r0, s0
; CHECK-SOFTFP-FULLFP16-NEXT:  mov       pc, lr

; CHECK-HARDFP-VFP3:  vmov r{{.}}, s0
; CHECK-HARDFP-VFP3:  vmov{{.*}}, s1
; CHECK-HARDFP-VFP3:  bl  __aeabi_h2f
; CHECK-HARDFP-VFP3:  bl  __aeabi_h2f
; CHECK-HARDFP-VFP3:  vadd.f32
; CHECK-HARDFP-VFP3:  bl  __aeabi_f2h
; CHECK-HARDFP-VFP3:  vmov  s0, r0

; CHECK-HARDFP-FP16:  vcvtb.f32.f16 [[S2:s[0-9]]], s1
; CHECK-HARDFP-FP16:  vcvtb.f32.f16 [[S0:s[0-9]]], s0
; CHECK-HARDFP-FP16:  vadd.f32  [[S0]], [[S0]], [[S2]]
; CHECK-HARDFP-FP16:  vcvtb.f16.f32 [[S0]], [[S0]]

; CHECK-HARDFP-FULLFP16:       vadd.f16  s0, s0, s1
; CHECK-HARDFP-FULLFP16-NEXT:  mov pc, lr
}
