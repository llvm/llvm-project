; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx2,fma -stop-after=expand-isel-pseudos 2>&1 | FileCheck %s

declare float @llvm.sqrt.f32(float) #0

define float @foo(float %f) #0 {
; CHECK: {{name: *foo}}
; CHECK: body:
; CHECK:     %0 = COPY %xmm0
; CHECK:     %1 = VRSQRTSSr killed %2, %0
; CHECK:     %3 = VMULSSrr %0, %1
; CHECK:     %4 = VMOVSSrm
; CHECK:     %5 = VFMADD213SSr %1, killed %3, %4
; CHECK:     %6 = VMOVSSrm
; CHECK:     %7 = VMULSSrr %1, %6
; CHECK:     %8 = VMULSSrr killed %7, killed %5
; CHECK:     %9 = VMULSSrr %0, %8
; CHECK:     %10 = VFMADD213SSr %8, %9, %4
; CHECK:     %11 = VMULSSrr %9, %6
; CHECK:     %12 = VMULSSrr killed %11, killed %10
; CHECK:     %14 = FsFLD0SS
; CHECK:     %15 = VCMPSSrr %0, killed %14, 0
; CHECK:     %17 = VANDNPSrr killed %16, killed %13
; CHECK:     %xmm0 = COPY %18
; CHECK:     RET 0, %xmm0
  %call = tail call float @llvm.sqrt.f32(float %f) #1
  ret float %call
}

define float @rfoo(float %f) #0 {
; CHECK: {{name: *rfoo}}
; CHECK: body:             |
; CHECK:     %0 = COPY %xmm0
; CHECK:     %1 = VRSQRTSSr killed %2, %0
; CHECK:     %3 = VMULSSrr %0, %1
; CHECK:     %4 = VMOVSSrm
; CHECK:     %5 = VFMADD213SSr %1, killed %3, %4
; CHECK:     %6 = VMOVSSrm
; CHECK:     %7 = VMULSSrr %1, %6
; CHECK:     %8 = VMULSSrr killed %7, killed %5
; CHECK:     %9 = VMULSSrr %0, %8
; CHECK:     %10 = VFMADD213SSr %8, killed %9, %4
; CHECK:     %11 = VMULSSrr %8, %6
; CHECK:     %12 = VMULSSrr killed %11, killed %10
; CHECK:     %xmm0 = COPY %12
; CHECK:     RET 0, %xmm0
  %sqrt = tail call float @llvm.sqrt.f32(float %f)
  %div = fdiv fast float 1.0, %sqrt
  ret float %div
}

attributes #0 = { "unsafe-fp-math"="true" "reciprocal-estimates"="sqrt:2" }
attributes #1 = { nounwind readnone }
