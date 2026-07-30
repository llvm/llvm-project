; RUN: llc < %s -mtriple=i686-- -mcpu=lakemont | FileCheck %s

; Make sure -mcpu=lakemont implies soft floats.
define float @test(float %a, float %b) nounwind readnone {
; CHECK-LABEL: test:
; CHECK: __addsf3
  %add = fadd float %a, %b
  ret float %add
}
