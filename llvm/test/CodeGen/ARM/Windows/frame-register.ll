; RUN: llc -mtriple thumbv7-windows -frame-pointer=all -filetype asm -o - %s \
; RUN:     | FileCheck %s

declare void @callee(i32)

define i32 @calleer(i32 %i) {
entry:
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %j, align 4
  %1 = load i32, ptr %j, align 4
  call void @callee(i32 %1)
  %2 = load i32, ptr %j, align 4
  %add1 = add nsw i32 %2, 1
  ret i32 %add1
}

; CHECK: push.w {r11, lr}

