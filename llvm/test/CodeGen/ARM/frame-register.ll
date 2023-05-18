; RUN: llc -mtriple arm-eabi -frame-pointer=all -filetype asm -o - %s \
; RUN:     | FileCheck -check-prefix CHECK-ARM %s

; RUN: llc -mtriple thumb-eabi -frame-pointer=all -filetype asm -o - %s \
; RUN:     | FileCheck -check-prefix CHECK-THUMB %s

; RUN: llc -mtriple arm-darwin -frame-pointer=all -filetype asm -o - %s \
; RUN:     | FileCheck -check-prefix CHECK-DARWIN-ARM %s

; RUN: llc -mtriple thumb-darwin -frame-pointer=all -filetype asm -o - %s \
; RUN:     | FileCheck -check-prefix CHECK-DARWIN-THUMB %s

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

; CHECK-ARM: push {r11, lr}
; CHECK-ARM: mov r11, sp

; CHECK-THUMB: push {r7, lr}
; CHECK-THUMB: add r7, sp, #0

; CHECK-DARWIN-ARM: push {r7, lr}
; CHECK-DARWIN-THUMB: push {r7, lr}

