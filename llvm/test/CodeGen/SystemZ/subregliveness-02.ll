; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z13 -systemz-subreg-liveness < %s | FileCheck %s

; Check for successful compilation.
; CHECK: meebr %f1, %f0

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-ibm-linux"

; Function Attrs: nounwind
define void @spec_random_load(i64 %a0) #0 {
bb:
  %tmp = sitofp i64 %a0 to float
  %tmp1 = fmul float %tmp, 0x3E00000000000000
  %tmp2 = fpext float %tmp1 to double
  %tmp3 = fmul double %tmp2, 2.560000e+02
  %tmp4 = fptosi double %tmp3 to i32
  %tmp5 = trunc i32 %tmp4 to i8
  store i8 %tmp5, i8* undef, align 1
  unreachable
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "stack-protector-buffer-size"="8" "target-cpu"="z13" "target-features"="+transactional-execution,+vector" "use-soft-float"="false" }
