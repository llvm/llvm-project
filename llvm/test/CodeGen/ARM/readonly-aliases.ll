; RUN: llc -mtriple thumbv7-unknown-linux-android -filetype asm -o - %s | FileCheck %s

@a = protected constant <{ i32, i32 }> <{ i32 0, i32 0 }>
@b = protected alias i32, getelementptr(i32, i32* getelementptr inbounds (<{ i32, i32 }>, <{ i32, i32 }>* @a, i32 0, i32 1), i32 -1)

declare void @f(i32*)

define void @g() {
entry:
  call void @f(i32* @b)
  ret void
}

; CHECK-LABEL: g:
; CHECK: movw [[REGISTER:r[0-9]+]], :lower16:b
; CHECK: movt [[REGISTER]], :upper16:b

