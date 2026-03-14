; RUN: llc -mtriple=thumbv7-windows -o - %s \
; RUN:   | FileCheck %s -check-prefix CHECK-WINDOWS

; RUN: llc -mtriple=thumbv7-eabi -o - %s \
; RUN:   | FileCheck %s -check-prefix CHECK-EABI

@i = common global i32 0, align 4
@j = common global i32 0, align 4

; Function Attrs: nounwind optsize readonly
define i32 @relocation(i32 %j, i32 %k) {
entry:
  %0 = load i32, ptr @i, align 4
  %1 = load i32, ptr @j, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

; CHECK-WINDOWS: movw r[[i:[0-4]]], :lower16:i
; CHECK-WINDOWS-NEXT: movt r[[i]], :upper16:i
; CHECK-WINDOWS: movw r[[j:[0-4]]], :lower16:j
; CHECK-WINDOWS-NEXT: movt r[[j]], :upper16:j

; CHECK-EABI: movw r[[i:[0-4]]], :lower16:i
; CHECK-EABI: movw r[[j:[0-4]]], :lower16:j
; CHECK-EABI-NEXT: movt r[[i]], :upper16:i
; CHECK-EABI-NEXT: movt r[[j]], :upper16:j
