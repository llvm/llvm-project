; Check that stubs generation for mips16 hard-float mode does not depend
; on the function 'use-soft-float' attribute's value.
; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel \
; RUN:     -mattr=mips16 -relocation-model=pic < %s | FileCheck %s

define void @bar_hf() #0 {
; CHECK: bar_hf:
entry:
  %call1 = call float @foo(float 1.000000e+00)
; CHECK: lw $2, %call16(foo)($3)
; CHECK: lw $5, %got(__mips16_call_stub_sf_1)($3)
  ret void
}

define void @bar_sf() #1 {
; CHECK: bar_sf:
entry:
  %call1 = call float @foo(float 1.000000e+00)
; CHECK: lw $3, %call16(foo)($2)
; CHECK-NOT: lw $5, %got(__mips16_call_stub_sf_1)($3)
  ret void
}

declare float @foo(float) #2

attributes #0 = {
  nounwind
  "less-precise-fpmad"="false" "frame-pointer"="all"
 "frame-pointer"="non-leaf" "no-infs-fp-math"="false"
  "no-nans-fp-math"="false" "stack-protector-buffer-size"="8"
  "unsafe-fp-math"="false" "use-soft-float"="false"
}
attributes #1 = {
  nounwind
  "less-precise-fpmad"="false" "frame-pointer"="all"
 "frame-pointer"="non-leaf" "no-infs-fp-math"="false"
  "no-nans-fp-math"="false" "stack-protector-buffer-size"="8"
  "unsafe-fp-math"="false" "use-soft-float"="true"
}
attributes #2 = {
  "less-precise-fpmad"="false" "frame-pointer"="all"
 "frame-pointer"="non-leaf" "no-infs-fp-math"="false"
  "no-nans-fp-math"="false" "stack-protector-buffer-size"="8"
  "unsafe-fp-math"="false" "use-soft-float"="true"
}
