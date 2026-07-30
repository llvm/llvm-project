; RUN: llc -mtriple armv7a--none-eabi -enable-ipra=false < %s | FileCheck %s
; RUN: llc -mtriple armv7a--none-eabi -enable-ipra=true  < %s | FileCheck %s

define i32 @returns_r0(i32 returned %a)  {
entry:
  call void asm sideeffect "", "~{r0}"()
  ret i32 %a
}

define i32 @test(i32 %a) {
; CHECK-LABEL: test:
entry:
; CHECK-NOT: r0
; CHECK: bl      returns_r0
; CHECK-NOT: r0
  %b = call i32 @returns_r0(i32 returned %a)
  ret i32 %a
}
