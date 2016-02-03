; RUN: llc -fp-contract=fast -O2 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-grtev4-linux-gnu"

; CHECK-LABEL: f
; CHECK-NOT: xsmaddmsp [[REG:[0-9]+]], [[REG]], {{[0-9]+}}
define float @f(float %xf) #0 {
  %1 = fmul float %xf, %xf
  %2 = fmul float %1, 0x3F43FB0140000000
  %3 = fsub float 1.000000e+00, %2
  %4 = fmul float %1, %3
  %5 = fmul float %4, 0x3F461C5440000000
  %6 = fsub float 1.000000e+00, %5
  %7 = fmul float %1, %6
  %8 = fmul float %7, 0x3F4899C100000000
  %9 = fsub float 1.000000e+00, %8
  %10 = fmul float %1, %9
  %11 = fmul float %10, 0x3F4B894020000000
  %12 = fsub float 1.000000e+00, %11
  %13 = fmul float %1, %12
  %14 = fmul float %13, 0x3F4F07C200000000
  %15 = fsub float 1.000000e+00, %14
  %16 = fmul float %1, %15
  %17 = fmul float %16, 0x3F519E0120000000
  %18 = fsub float 1.000000e+00, %17
  %19 = fmul float %1, %18
  %20 = fmul float %19, 0x3F542D6620000000
  %21 = fsub float 1.000000e+00, %20
  %22 = fmul float %1, %21
  %23 = fmul float %22, 0x3F5756CAC0000000
  %24 = fsub float 1.000000e+00, %23
  %25 = fmul float %1, %24
  ret float %25
}

attributes #0 = { norecurse nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pwr8" "target-features"="+altivec,+bpermd,+crypto,+direct-move,+extdiv,+power8-vector,+vsx,-qpx" "unsafe-fp-math"="false" "use-soft-float"="false" }
