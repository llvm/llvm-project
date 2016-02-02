; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+f16c < %s | FileCheck %s --check-prefix=ALL --check-prefix=F16C
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+avx < %s | FileCheck %s --check-prefix=ALL --check-prefix=AVX

define zeroext i16 @test1_fast(double %d) #0 {
; ALL-LABEL: test1_fast:
; F16C-NOT: callq {{_+}}truncdfhf2
; F16C: vcvtsd2ss %xmm0, %xmm0, %xmm0
; F16C-NEXT: vcvtps2ph $4, %xmm0, %xmm0
; AVX: callq {{_+}}truncdfhf2
; ALL: ret
entry:
  %0 = tail call i16 @llvm.convert.to.fp16.f64(double %d)
  ret i16 %0
}

define zeroext i16 @test2_fast(x86_fp80 %d) #0 {
; ALL-LABEL: test2_fast:
; F16C-NOT: callq {{_+}}truncxfhf2
; F16C: fldt
; F16C-NEXT: fstps
; F16C-NEXT: vmovss
; F16C-NEXT: vcvtps2ph $4, %xmm0, %xmm0
; AVX: callq {{_+}}truncxfhf2
; ALL: ret
entry:
  %0 = tail call i16 @llvm.convert.to.fp16.f80(x86_fp80 %d)
  ret i16 %0
}

define zeroext i16 @test1(double %d) #1 {
; ALL-LABEL: test1:
; ALL: callq  {{_+}}truncdfhf2
; ALL: ret
entry:
  %0 = tail call i16 @llvm.convert.to.fp16.f64(double %d)
  ret i16 %0
}

define zeroext i16 @test2(x86_fp80 %d) #1 {
; ALL-LABEL: test2:
; ALL: callq  {{_+}}truncxfhf2
; ALL: ret
entry:
  %0 = tail call i16 @llvm.convert.to.fp16.f80(x86_fp80 %d)
  ret i16 %0
}

declare i16 @llvm.convert.to.fp16.f64(double)
declare i16 @llvm.convert.to.fp16.f80(x86_fp80)

attributes #0 = { nounwind readnone uwtable "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind readnone uwtable "unsafe-fp-math"="false" "use-soft-float"="false" }
