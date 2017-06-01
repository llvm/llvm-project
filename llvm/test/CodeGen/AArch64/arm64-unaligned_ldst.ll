; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s
; rdar://r11231896

define void @t1(i8* nocapture %a, i8* nocapture %b) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK-NOT: orr
; CHECK: ldr [[X0:x[0-9]+]], [x1]
; CHECK: str [[X0]], [x0]
  %tmp1 = bitcast i8* %b to i64*
  %tmp2 = bitcast i8* %a to i64*
  %tmp3 = load i64, i64* %tmp1, align 1
  store i64 %tmp3, i64* %tmp2, align 1
  ret void
}

define void @t2(i8* nocapture %a, i8* nocapture %b) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK-NOT: orr
; CHECK: ldr [[W0:w[0-9]+]], [x1]
; CHECK: str [[W0]], [x0]
  %tmp1 = bitcast i8* %b to i32*
  %tmp2 = bitcast i8* %a to i32*
  %tmp3 = load i32, i32* %tmp1, align 1
  store i32 %tmp3, i32* %tmp2, align 1
  ret void
}

define void @t3(i8* nocapture %a, i8* nocapture %b) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK-NOT: orr
; CHECK: ldrh [[W0:w[0-9]+]], [x1]
; CHECK: strh [[W0]], [x0]
  %tmp1 = bitcast i8* %b to i16*
  %tmp2 = bitcast i8* %a to i16*
  %tmp3 = load i16, i16* %tmp1, align 1
  store i16 %tmp3, i16* %tmp2, align 1
  ret void
}
