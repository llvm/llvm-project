; RUN: llc -mtriple=hexagon < %s | FileCheck %s

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
;CHECK: jump __extendhfsf2
;CHECK: r0 = memuh
define dso_local float @test1(ptr readonly captures(none) %a) local_unnamed_addr #0 {
entry:
  %0 = load i16, ptr %a, align 2
  %1 = bitcast i16 %0 to half
  %2 = fpext half %1 to float
  ret float %2
}

;CHECK-LABEL: @test2
;CHECK: call __extendhfsf2
;CHECK: r0 = memuh
;CHECK: convert_sf2d
define dso_local double @test2(ptr readonly captures(none) %a) local_unnamed_addr #0 {
entry:
  %0 = load i16, ptr %a, align 2
  %1 = bitcast i16 %0 to half
  %2 = fpext half %1 to double
  ret double %2
}

;CHECK-LABEL: @test3
;CHECK: call __truncsfhf2
;CHECK: memh{{.*}}= r0
define dso_local void @test3(float %src, ptr captures(none) %dst) local_unnamed_addr #0 {
entry:
  %0 = fptrunc float %src to half
  %1 = bitcast half %0 to i16
  store i16 %1, ptr %dst, align 2
  ret void
}

;CHECK-LABEL: @test4
;CHECK: call __truncdfhf2
;CHECK: memh{{.*}}= r0
define dso_local void @test4(double %src, ptr captures(none) %dst) local_unnamed_addr #0 {
entry:
  %0 = fptrunc double %src to half
  %1 = bitcast half %0 to i16
  store i16 %1, ptr %dst, align 2
  ret void
}

;CHECK-LABEL: @test5
;CHECK: call __extendhfsf2
;CHECK: call __extendhfsf2
;CHECK: sfadd
define dso_local float @test5(ptr readonly captures(none) %a, ptr readonly captures(none) %b) local_unnamed_addr #0 {
entry:
  %0 = load i16, ptr %a, align 2
  %1 = bitcast i16 %0 to half
  %2 = fpext half %1 to float
  %3 = load i16, ptr %b, align 2
  %4 = bitcast i16 %3 to half
  %5 = fpext half %4 to float
  %add = fadd float %2, %5
  ret float %add
}

attributes #0 = { nounwind readonly }
