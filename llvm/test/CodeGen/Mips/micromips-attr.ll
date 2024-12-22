; RUN: llc -march=mips -mcpu=mips32 --mattr=-micromips < %s | FileCheck %s 

define void @foo() #0 {
entry:
  ret void
}
; CHECK:        .set    micromips
; CHECK-NEXT:   .set    nomips16
; CHECK-NEXT:   .ent    foo
; CHECK-NEXT: foo:

define void @bar() #1 {
entry:
  ret void
}
; CHECK:        .set    nomicromips
; CHECK-NEXT:   .set    nomips16
; CHECK-NEXT:   .ent    bar
; CHECK-NEXT: bar:

attributes #0 = {
  nounwind "micromips"
  "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false"
  "less-precise-fpmad"="false" "frame-pointer"="none"
  "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false"
  "no-signed-zeros-fp-math"="false" "no-trapping-math"="false"
  "stack-protector-buffer-size"="8" "unsafe-fp-math"="false"
  "use-soft-float"="false"
}

attributes #1 = {
  nounwind
  "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false"
  "less-precise-fpmad"="false" "frame-pointer"="none"
  "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false"
  "no-signed-zeros-fp-math"="false" "no-trapping-math"="false"
  "stack-protector-buffer-size"="8" "unsafe-fp-math"="false"
  "use-soft-float"="false"
}
