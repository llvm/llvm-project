; RUN: opt -mem2reg < %s -S | FileCheck %s


; CHECK-LABEL: void @patatino()
; CHECK-NEXT: ret void

; CHECK-LABEL: cantreachme:
; CHECK-NEXT: %dec = add nsw i32 undef, -1
; CHECK-NEXT: br label %cantreachme

define void @patatino() {
  %a = alloca i32, align 4
  ret void
cantreachme:
  %dec = add nsw i32 %tmp, -1
  store i32 %dec, i32* %a
  store i32 %tmp, i32* %a
  %tmp = load i32, i32* %a
  br label %cantreachme
}
