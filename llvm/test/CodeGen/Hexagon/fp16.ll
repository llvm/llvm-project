; RUN: llc -march=hexagon < %s | FileCheck %s

; This test validates the following facts for half-precision floating point
; conversions.
; Generate correct libcall names for conversion from fp16 to fp32.
; (__extendhfsf2).
;  The extension from fp16 to fp64 is implicitly handled by __extendhfsf2 and convert_sf2d.
; (fp16->fp32->fp64).
; Generate correcct libcall names for conversion from fp32/fp64 to fp16
; (__truncsfhf2 and __truncdfhf2)
; Verify that we generate loads and stores of halfword.

; Validate that we generate correct lib calls to convert fp16

;CHECK-LABEL: @test1
;CHECK: call __extendhfsf2
;CHECK: r0 = memuh
define dso_local float @test1(i16* nocapture readonly %a) local_unnamed_addr #0 {
entry:
  %0 = load i16, i16* %a, align 2
  %1 = tail call float @llvm.convert.from.fp16.f32(i16 %0)
  ret float %1
}

;CHECK-LABEL: @test2
;CHECK: call __extendhfsf2
;CHECK: r0 = memuh
;CHECK: convert_sf2d
define dso_local double @test2(i16* nocapture readonly %a) local_unnamed_addr #0 {
entry:
  %0 = load i16, i16* %a, align 2
  %1 = tail call double @llvm.convert.from.fp16.f64(i16 %0)
  ret double %1
}

;CHECK-LABEL: @test3
;CHECK: call __truncsfhf2
;CHECK: memh{{.*}}= r0
define dso_local void @test3(float %src, i16* nocapture %dst) local_unnamed_addr #0 {
entry:
  %0 = tail call i16 @llvm.convert.to.fp16.f32(float %src)
  store i16 %0, i16* %dst, align 2
  ret void
}

;CHECK-LABEL: @test4
;CHECK: call __truncdfhf2
;CHECK: memh{{.*}}= r0
define dso_local void @test4(double %src, i16* nocapture %dst) local_unnamed_addr #0 {
entry:
  %0 = tail call i16 @llvm.convert.to.fp16.f64(double %src)
  store i16 %0, i16* %dst, align 2
  ret void
}

;CHECK-LABEL: @test5
;CHECK: call __extendhfsf2
;CHECK: call __extendhfsf2
;CHECK: sfadd
define dso_local float @test5(i16* nocapture readonly %a, i16* nocapture readonly %b) local_unnamed_addr #0 {
entry:
  %0 = load i16, i16* %a, align 2
  %1 = tail call float @llvm.convert.from.fp16.f32(i16 %0)
  %2 = load i16, i16* %b, align 2
  %3 = tail call float @llvm.convert.from.fp16.f32(i16 %2)
  %add = fadd float %1, %3
  ret float %add
}

declare float @llvm.convert.from.fp16.f32(i16) #1
declare double @llvm.convert.from.fp16.f64(i16) #1
declare i16 @llvm.convert.to.fp16.f32(float) #1
declare i16 @llvm.convert.to.fp16.f64(double) #1

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind readnone }
