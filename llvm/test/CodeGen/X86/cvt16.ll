; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=-f16c | FileCheck %s -check-prefix=CHECK -check-prefix=LIBCALL
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+f16c | FileCheck %s -check-prefix=CHECK -check-prefix=F16C
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=-f16c,+soft-float | FileCheck %s -check-prefix=CHECK -check-prefix=SOFTFLOAT
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+f16c,+soft-float | FileCheck %s -check-prefix=CHECK -check-prefix=SOFTFLOAT

; This is a test for float to half float conversions on x86-64.
;
; If flag -soft-float is set, or if there is no F16C support, then:
; 1) half float to float conversions are
;    translated into calls to __gnu_h2f_ieee defined
;    by the compiler runtime library;
; 2) float to half float conversions are translated into calls
;    to __gnu_f2h_ieee which expected to be defined by the
;    compiler runtime library.
;
; Otherwise (we have F16C support):
; 1) half float to float conversion are translated using
;    vcvtph2ps instructions;
; 2) float to half float conversions are translated using
;    vcvtps2ph instructions


define void @test1(float %src, i16* %dest) {
  %1 = tail call i16 @llvm.convert.to.fp16.f32(float %src)
  store i16 %1, i16* %dest, align 2
  ret void
}
; CHECK-LABEL: test1
; LIBCALL: callq  __gnu_f2h_ieee
; SOFTFLOAT: callq  __gnu_f2h_ieee
; F16C: vcvtps2ph
; CHECK: ret


define float @test2(i16* nocapture %src) {
  %1 = load i16, i16* %src, align 2
  %2 = tail call float @llvm.convert.from.fp16.f32(i16 %1)
  ret float %2
}
; CHECK-LABEL: test2:
; LIBCALL: jmp  __gnu_h2f_ieee
; SOFTFLOAT: callq  __gnu_h2f_ieee
; F16C: vcvtph2ps
; F16C: ret


define float @test3(float %src) nounwind uwtable readnone {
  %1 = tail call i16 @llvm.convert.to.fp16.f32(float %src)
  %2 = tail call float @llvm.convert.from.fp16.f32(i16 %1)
  ret float %2
}

; CHECK-LABEL: test3:
; LIBCALL: callq  __gnu_f2h_ieee
; LIBCALL: jmp   __gnu_h2f_ieee
; SOFTFLOAT: callq  __gnu_f2h_ieee
; SOFTFLOAT: callq  __gnu_h2f_ieee
; F16C: vcvtps2ph
; F16C-NEXT: vcvtph2ps
; F16C: ret

define double @test4(i16* nocapture %src) {
  %1 = load i16, i16* %src, align 2
  %2 = tail call double @llvm.convert.from.fp16.f64(i16 %1)
  ret double %2
}
; CHECK-LABEL: test4:
; LIBCALL: callq  __gnu_h2f_ieee
; LIBCALL: cvtss2sd
; SOFTFLOAT: callq  __gnu_h2f_ieee
; SOFTFLOAT: callq __extendsfdf2
; F16C: vcvtph2ps
; F16C: vcvtss2sd
; F16C: ret


define i16 @test5(double %src) {
  %val = tail call i16 @llvm.convert.to.fp16.f64(double %src)
  ret i16 %val
}
; CHECK-LABEL: test5:
; LIBCALL: jmp  __truncdfhf2
; SOFTFLOAT: callq  __truncdfhf2
; F16C: jmp __truncdfhf2

declare float @llvm.convert.from.fp16.f32(i16) nounwind readnone
declare i16 @llvm.convert.to.fp16.f32(float) nounwind readnone
declare double @llvm.convert.from.fp16.f64(i16) nounwind readnone
declare i16 @llvm.convert.to.fp16.f64(double) nounwind readnone
