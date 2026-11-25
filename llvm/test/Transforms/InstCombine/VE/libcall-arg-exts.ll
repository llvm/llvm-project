; RUN: opt < %s -passes=instcombine -S -mtriple=ve | FileCheck %s

; Check that i32 parameters of generated libcalls have the proper extension
; attributes.

declare double @exp2(double)
declare i32 @fputs(ptr, ptr)

@A = constant [2 x i8] c"A\00"

define double @exp2_to_ldexp(i32 %x) {
; CHECK-LABEL: define double @exp2_to_ldexp(
; CHECK:         call double @ldexp(double 1.000000e+00, i32 %x)
  %conv = sitofp i32 %x to double
  %ret = call double @exp2(double %conv)
  ret double %ret
}

define void @fputs_to_fputc(ptr %fp) {
; CHECK-LABEL: define void @fputs_to_fputc(
; CHECK:         call i32 @fputc(i32 65, ptr %fp)
  call i32 @fputs(ptr @A, ptr %fp)
  ret void
}

; CHECK: declare double @ldexp(double, i32 signext)
; CHECK: declare noundef i32 @fputc(i32 noundef signext, ptr noundef captures(none))
