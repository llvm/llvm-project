; RUN: llc -mtriple=hexagon-unknown-linux-musl < %s | FileCheck %s

define void @Foo(i32 signext %a, i32 signext %b) #0 {
; CHECK-LABEL: @Foo
; CHECK-LABEL: .Lxray_sled_0:
; CHECK:        jump .Ltmp0
; CHECK-COUNT-4: nop
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %return, label %if.end

; CHECK-LABEL: .Lxray_sled_1:
; CHECK:        jump .Ltmp1
; CHECK-COUNT-4: nop
; CHECK-LABEL: .Ltmp1:
; CHECK:       if (p0) jumpr:nt r31
if.end:
  tail call void @Bar()
  br label %return

return:
  ret void
}

declare void @Bar()

attributes #0 = { "function-instrument"="xray-always" }
