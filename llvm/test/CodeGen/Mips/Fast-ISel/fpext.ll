; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@f = global float 0x40147E6B80000000, align 4
@d_f = common global double 0.000000e+00, align 8
@.str = private unnamed_addr constant [6 x i8] c"%f  \0A\00", align 1

; Function Attrs: nounwind
define void @dv() #0 {
entry:
  %0 = load float, ptr @f, align 4
  %conv = fpext float %0 to double
; CHECK: cvt.d.s  $f{{[0-9]+}}, $f{{[0-9]+}}
  store double %conv, ptr @d_f, align 8
  ret void
}


attributes #1 = { nounwind }
