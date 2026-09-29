; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: r0 = sub(#0,r0)
define i32 @f0(i32 %a0) #0 {
  %v0 = sub i32 0, %a0
  ret i32 %v0
}

; CHECK-LABEL: f1:
; CHECK: r1:0 = neg(r1:0)
define i64 @f1(i64 %a0) #0 {
  %v0 = sub i64 0, %a0
  ret i64 %v0
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" }
