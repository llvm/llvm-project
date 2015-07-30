; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=1 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=1 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s


@f = global float 0x40D6E83280000000, align 4
@d = global double 0x4132D68780000000, align 8
@i_f = common global i32 0, align 4
@i_d = common global i32 0, align 4
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

; Function Attrs: nounwind
define void @ifv() {
entry:
; CHECK-LABEL:   .ent  ifv
  %0 = load float, float* @f, align 4
  %conv = fptosi float %0 to i32
; CHECK:   trunc.w.s  $f[[REG:[0-9]+]], $f{{[0-9]+}}
; CHECK:   mfc1	${{[0-9]+}}, $f[[REG]]
  store i32 %conv, i32* @i_f, align 4
  ret void
}

; Function Attrs: nounwind
define void @idv() {
entry:
; CHECK-LABEL:   .ent  idv
  %0 = load double, double* @d, align 8
  %conv = fptosi double %0 to i32
; CHECK:   trunc.w.d  $f[[REG:[0-9]+]], $f{{[0-9]+}}
; CHECK:   mfc1	${{[0-9]+}}, $f[[REG]]
  store i32 %conv, i32* @i_d, align 4
  ret void
}
