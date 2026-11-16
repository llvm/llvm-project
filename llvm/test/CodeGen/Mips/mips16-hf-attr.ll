; Check that MIPS16 hard-float helpers follow each function's attributes,
; including when the default ISA is MIPS32.
; RUN: llc -mtriple=mipsel-linux-gnu \
; RUN:     -mattr=mips16 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-linux-gnu -relocation-model=pic < %s | FileCheck %s

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

define float @return_hf(float %x) #0 {
; CHECK-LABEL: return_hf:
; CHECK: %call16(__mips16_ret_sf)
; CHECK: .end return_hf
  ret float %x
}

define float @return_sf(float %x) #1 {
; CHECK-LABEL: return_sf:
; CHECK-NOT: __mips16_ret_sf
; CHECK: .end return_sf
  ret float %x
}

define float @return_sf32(float %x) "nomips16" "use-soft-float"="true" {
; CHECK-LABEL: return_sf32:
; CHECK: move $2, $4
; CHECK: .end return_sf32
  ret float %x
}

; CHECK: .section .mips16.fn.return_hf,
; CHECK-LABEL: __fn_stub_return_hf:
; CHECK-NOT: .mips16.fn.return_sf

declare float @foo(float) #2

attributes #0 = {
  nounwind
  "mips16"
  "less-precise-fpmad"="false" "frame-pointer"="all"
 "frame-pointer"="non-leaf" "no-infs-fp-math"="false"
  "no-nans-fp-math"="false" "stack-protector-buffer-size"="8"
  "use-soft-float"="false"
}
attributes #1 = {
  nounwind
  "mips16"
  "less-precise-fpmad"="false" "frame-pointer"="all"
 "frame-pointer"="non-leaf" "no-infs-fp-math"="false"
  "no-nans-fp-math"="false" "stack-protector-buffer-size"="8"
  "use-soft-float"="true"
}
attributes #2 = {
  "less-precise-fpmad"="false" "frame-pointer"="all"
 "frame-pointer"="non-leaf" "no-infs-fp-math"="false"
  "no-nans-fp-math"="false" "stack-protector-buffer-size"="8"
  "use-soft-float"="true"
}
