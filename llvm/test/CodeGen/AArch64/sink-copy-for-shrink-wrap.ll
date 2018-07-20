; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s

; CHECK-LABEL: %bb.0:
; CHECK-NOT: stp
; CHECK-NOT: mov w{{[0-9]+}}, w0
; CHECK-LABEL: %bb.1:
; CHECK: stp x19
; CHECK: mov w{{[0-9]+}}, w0

define i32 @shrinkwrapme(i32 %paramAcrossCall, i32 %paramNotAcrossCall) {
entry:
  %cmp5 = icmp sgt i32 %paramNotAcrossCall, 0
  br i1 %cmp5, label %CallBB, label %Exit
CallBB:
  %call = call i32 @fun()
  %add = add i32 %call, %paramAcrossCall
  ret i32 %add
Exit:
  ret i32 0
}

declare i32 @fun()
