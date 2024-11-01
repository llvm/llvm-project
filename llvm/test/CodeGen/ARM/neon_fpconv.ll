; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

; PR12540: ARM backend lowering of FP_ROUND v2f64 to v2f32.
define <2 x float> @vtrunc(<2 x double> %a) {
; CHECK: vcvt.f32.f64 [[S0:s[0-9]+]], [[D0:d[0-9]+]]
; CHECK: vcvt.f32.f64 [[S1:s[0-9]+]], [[D1:d[0-9]+]]
  %vt = fptrunc <2 x double> %a to <2 x float>
  ret <2 x float> %vt
}

define <2 x double> @vextend(<2 x float> %a) {
; CHECK: vcvt.f64.f32 [[D0:d[0-9]+]], [[S0:s[0-9]+]]
; CHECK: vcvt.f64.f32 [[D1:d[0-9]+]], [[S1:s[0-9]+]]
  %ve = fpext <2 x float> %a to <2 x double>
  ret <2 x double> %ve
}

; We used to generate vmovs between scalar and vfp/neon registers.
; CHECK: vsitofp_double
define void @vsitofp_double(ptr %loadaddr,
                            ptr %storeaddr) {
  %v0 = load <2 x i32>, ptr %loadaddr
; CHECK:      vldr
; CHECK-NEXT:	vcvt.f64.s32
; CHECK-NEXT:	vcvt.f64.s32
; CHECK-NEXT:	vst
  %r = sitofp <2 x i32> %v0 to <2 x double>
  store <2 x double> %r, ptr %storeaddr
  ret void
}
; CHECK: vuitofp_double
define void @vuitofp_double(ptr %loadaddr,
                            ptr %storeaddr) {
  %v0 = load <2 x i32>, ptr %loadaddr
; CHECK:      vldr
; CHECK-NEXT:	vcvt.f64.u32
; CHECK-NEXT:	vcvt.f64.u32
; CHECK-NEXT:	vst
  %r = uitofp <2 x i32> %v0 to <2 x double>
  store <2 x double> %r, ptr %storeaddr
  ret void
}
