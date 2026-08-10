; RUN: llc -mtriple=thumbv8m.main -mcpu=cortex-m33 --float-abi=hard %s -o - | \
; RUN:   FileCheck %s

@g = hidden local_unnamed_addr global ptr null, align 4
@a = hidden local_unnamed_addr global float 0.000000e+00, align 4

define hidden void @f() local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr @g, align 4
  %call = tail call nnan ninf nsz float %0() #1
  store float %call, ptr @a, align 4
  ret void
}

; CHECK: blxns r{{[0-9]+}}
; CHECK: vmov  r[[T:[0-9]+]], s0
; CHECK: vlldm sp
; CHECK: vmov  s0, r[[T]]

attributes #0 = { nounwind }
attributes #1 = { nounwind "cmse_nonsecure_call" }
