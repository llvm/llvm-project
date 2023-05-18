; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s
; rdar://r11231896

define void @t1(ptr nocapture %a, ptr nocapture %b) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK-NOT: orr
; CHECK: ldr [[X0:x[0-9]+]], [x1]
; CHECK: str [[X0]], [x0]
  %tmp3 = load i64, ptr %b, align 1
  store i64 %tmp3, ptr %a, align 1
  ret void
}

define void @t2(ptr nocapture %a, ptr nocapture %b) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK-NOT: orr
; CHECK: ldr [[W0:w[0-9]+]], [x1]
; CHECK: str [[W0]], [x0]
  %tmp3 = load i32, ptr %b, align 1
  store i32 %tmp3, ptr %a, align 1
  ret void
}

define void @t3(ptr nocapture %a, ptr nocapture %b) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK-NOT: orr
; CHECK: ldrh [[W0:w[0-9]+]], [x1]
; CHECK: strh [[W0]], [x0]
  %tmp3 = load i16, ptr %b, align 1
  store i16 %tmp3, ptr %a, align 1
  ret void
}
