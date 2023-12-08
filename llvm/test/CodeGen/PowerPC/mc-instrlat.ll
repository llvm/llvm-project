; RUN: llc -verify-machineinstrs -O3 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(double %eps) #0 {
entry:
  %0 = fmul fast double %eps, %eps
  %div = fmul fast double %0, 0x3FD5555555555555
  tail call void @bar(double %div) #2
  unreachable

; This used to crash because we'd call a function to compute instruction
; latency not supported with itineraries.
; CHECK-LABEL: @foo
; CHECK: bar

}

declare void @bar(double) #1

attributes #0 = { nounwind "no-infs-fp-math"="true" "no-nans-fp-math"="true" "target-cpu"="ppc64" "target-features"="+altivec,-bpermd,-crypto,-direct-move,-extdiv,-power8-vector,-vsx" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="ppc64" "target-features"="+altivec,-bpermd,-crypto,-direct-move,-extdiv,-power8-vector,-vsx" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }

