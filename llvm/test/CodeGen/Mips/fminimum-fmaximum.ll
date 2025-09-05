; RUN: llc -mtriple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64r2 -mattr=+nan2008 < %s | FileCheck -check-prefixes=NAN2008 %s
; RUN: llc -mtriple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64r2 -mattr=-nan2008 < %s | FileCheck -check-prefixes=LEGACY %s

define float @test_fminimum_f32(float %a, float %b) {
; NAN2008:    .4byte 0x7fc00000    # float NaN
;
; LEGACY:    .4byte 0x7fbfffff    # float NaN
  %val = tail call float @llvm.minimum.f32(float %a, float %b)
  ret float %val
}

define double @test_fminimum_f64(double %a, double %b) {
; NAN2008:    .8byte 0x7ff8000000000000    # double NaN
;
; LEGACY:    .8byte 0x7ff7ffffffffffff    # double NaN
  %val = tail call double @llvm.minimum.f64(double %a, double %b)
  ret double %val
}
