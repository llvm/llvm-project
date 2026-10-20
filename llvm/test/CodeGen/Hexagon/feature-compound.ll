; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: t0
; CHECK: r0 = add(r1,add(r0,#23))
define i32 @t0(i32 %a0, i32 %a1) #0 {
  %v0 = add i32 %a1, 23
  %v1 = add i32 %a0, %v0
  ret i32 %v1
}

; CHECK-LABEL: t1
; CHECK: r[[R:[0-9]+]] = add(r1,r0)
; CHECK: r0 = add(r[[R]],#23)
define i32 @t1(i32 %a0, i32 %a1) #1 {
  %v0 = add i32 %a1, 23
  %v1 = add i32 %a0, %v0
  ret i32 %v1
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv62" "target-features"="+compound" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv62" "target-features"="-compound" }
