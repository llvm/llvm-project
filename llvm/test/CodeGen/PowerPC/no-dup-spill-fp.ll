; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64"

; Function Attrs: nounwind
define void @test() #0 {
entry:
  call void @func()
  call void asm sideeffect "nop", "~{r31}"() #1, !srcloc !0
  ret void

; CHECK-LABEL: @test
; CHECK: std 31, -8(1)
; CHECK: stdu 1, -{{[0-9]+}}(1)
; CHECK-NOT: std 31,
; CHECK: bl func
; CHECK: ld 31, -8(1)
; CHECK: blr
}

declare void @func()

attributes #0 = { nounwind "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "target-cpu"="ppc64" }
attributes #1 = { nounwind }

!0 = !{i32 57}
