; RUN: opt < %s -instsimplify -S | FileCheck %s

; PR8862

; CHECK: @shift1
; CHECK: ret i32 %A
define i32 @shift1(i32 %A, i32 %B) {
  %C = lshr exact i32 %A, %B
  %D = shl nuw i32 %C, %B
  ret i32 %D
}

; CHECK: @shift2
; CHECK: lshr
; CHECK: ret i32 %D
define i32 @shift2(i32 %A, i32 %B) {
  %C = lshr i32 %A, %B
  %D = shl nuw i32 %C, %B
  ret i32 %D
}

; CHECK: @shift3
; CHECK: ret i32 %A
define i32 @shift3(i32 %A, i32 %B) {
  %C = ashr exact i32 %A, %B
  %D = shl nuw i32 %C, %B
  ret i32 %D
}

; CHECK: @shift4
; CHECK: ret i32 %A
define i32 @shift4(i32 %A, i32 %B) {
  %C = shl nuw i32 %A, %B
  %D = lshr i32 %C, %B
  ret i32 %D
}

; CHECK: @shift5
; CHECK: ret i32 %A
define i32 @shift5(i32 %A, i32 %B) {
  %C = shl nsw i32 %A, %B
  %D = ashr i32 %C, %B
  ret i32 %D
}
