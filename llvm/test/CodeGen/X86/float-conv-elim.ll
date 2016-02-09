; RUN: llc -march=x86-64 -mcpu=x86-64 < %s | FileCheck %s

; Make sure the float conversion is folded away as it should be.
; CHECK-LABEL: foo
; CHECK-NOT: cvt
; CHECK: movzbl
define i32 @foo(i8 %a) #0 {
  %conv = uitofp i8 %a to float
  %conv1 = fptosi float %conv to i32
  ret i32 %conv1
}

; CHECK-LABEL: foo2
; CHECK-NOT: cvt
; CHECK: movsbl
define i32 @foo2(i8 %a) #0 {
  %conv = sitofp i8 %a to float
  %conv1 = fptosi float %conv to i32
  ret i32 %conv1
}

; CHECK-LABEL: bar
; CHECK-NOT: cvt
; CHECK: movb
define zeroext i8 @bar(i8 zeroext %a) #0 {
  %conv = uitofp i8 %a to float
  %conv1 = fptoui float %conv to i8
  ret i8 %conv1
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

