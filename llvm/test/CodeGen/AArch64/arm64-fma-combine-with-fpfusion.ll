; RUN: llc < %s -mtriple=aarch64-linux-gnu -fp-contract=fast | FileCheck %s
define float @mul_add(float %a, float %b, float %c) local_unnamed_addr #0 {
; CHECK-LABEL: %entry
; CHECK: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  entry:
    %mul = fmul float %a, %b
    %add = fadd float %mul, %c
    ret float %add
}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }

