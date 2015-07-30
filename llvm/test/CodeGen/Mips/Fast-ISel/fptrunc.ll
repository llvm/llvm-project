; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=1 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=1 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@d = global double 0x40147E6B74DF0446, align 8
@f = common global float 0.000000e+00, align 4
@.str = private unnamed_addr constant [6 x i8] c"%f  \0A\00", align 1

; Function Attrs: nounwind
define void @fv() #0 {
entry:
  %0 = load double, double* @d, align 8
  %conv = fptrunc double %0 to float
; CHECK: cvt.s.d  $f{{[0-9]+}}, $f{{[0-9]+}}
  store float %conv, float* @f, align 4
  ret void
}

attributes #1 = { nounwind }
