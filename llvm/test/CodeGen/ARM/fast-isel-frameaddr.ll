; RUN: llc < %s -O0 -fast-isel-abort -mtriple=armv7-apple-darwin | FileCheck %s --check-prefix=DARWIN-ARM
; RUN: llc < %s -O0 -fast-isel-abort -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=LINUX-ARM
; RUN: llc < %s -O0 -fast-isel-abort -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=DARWIN-THUMB2
; RUN: llc < %s -O0 -fast-isel-abort -mtriple=thumbv7-linux-gnueabi | FileCheck %s --check-prefix=LINUX-THUMB2

define i8* @frameaddr_index0() nounwind {
entry:
; DARWIN-ARM: frameaddr_index0:
; DARWIN-ARM: push {r7}
; DARWIN-ARM: mov r7, sp
; DARWIN-ARM: mov r0, r7

; DARWIN-THUMB2: frameaddr_index0:
; DARWIN-THUMB2: str r7, [sp, #-4]!
; DARWIN-THUMB2: mov r7, sp
; DARWIN-THUMB2: mov r0, r7

; LINUX-ARM: frameaddr_index0:
; LINUX-ARM: push {r11}
; LINUX-ARM: mov r11, sp
; LINUX-ARM: mov r0, r11

; LINUX-THUMB2: frameaddr_index0:
; LINUX-THUMB2: str r7, [sp, #-4]!
; LINUX-THUMB2: mov r7, sp
; LINUX-THUMB2: mov r0, r7

  %0 = call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @frameaddr_index1() nounwind {
entry:
; DARWIN-ARM: frameaddr_index1:
; DARWIN-ARM: push {r7}
; DARWIN-ARM: mov r7, sp
; DARWIN-ARM: mov r0, r7
; DARWIN-ARM: ldr r0, [r0]

; DARWIN-THUMB2: frameaddr_index1:
; DARWIN-THUMB2: str r7, [sp, #-4]!
; DARWIN-THUMB2: mov r7, sp
; DARWIN-THUMB2: mov r0, r7
; DARWIN-THUMB2: ldr r0, [r0]

; LINUX-ARM: frameaddr_index1:
; LINUX-ARM: push {r11}
; LINUX-ARM: mov r11, sp
; LINUX-ARM: mov r0, r11
; LINUX-ARM: ldr r0, [r0]

; LINUX-THUMB2: frameaddr_index1:
; LINUX-THUMB2: str r7, [sp, #-4]!
; LINUX-THUMB2: mov r7, sp
; LINUX-THUMB2: mov r0, r7
; LINUX-THUMB2: ldr r0, [r0]

  %0 = call i8* @llvm.frameaddress(i32 1)
  ret i8* %0
}

define i8* @frameaddr_index3() nounwind {
entry:
; DARWIN-ARM: frameaddr_index3:
; DARWIN-ARM: push {r7}
; DARWIN-ARM: mov r7, sp
; DARWIN-ARM: mov r0, r7
; DARWIN-ARM: ldr r0, [r0]
; DARWIN-ARM: ldr r0, [r0]
; DARWIN-ARM: ldr r0, [r0]

; DARWIN-THUMB2: frameaddr_index3:
; DARWIN-THUMB2: str r7, [sp, #-4]!
; DARWIN-THUMB2: mov r7, sp
; DARWIN-THUMB2: mov r0, r7
; DARWIN-THUMB2: ldr r0, [r0]
; DARWIN-THUMB2: ldr r0, [r0]
; DARWIN-THUMB2: ldr r0, [r0]

; LINUX-ARM: frameaddr_index3:
; LINUX-ARM: push {r11}
; LINUX-ARM: mov r11, sp
; LINUX-ARM: mov r0, r11
; LINUX-ARM: ldr r0, [r0]
; LINUX-ARM: ldr r0, [r0]
; LINUX-ARM: ldr r0, [r0]

; LINUX-THUMB2: frameaddr_index3:
; LINUX-THUMB2: str r7, [sp, #-4]!
; LINUX-THUMB2: mov r7, sp
; LINUX-THUMB2: mov r0, r7
; LINUX-THUMB2: ldr r0, [r0]
; LINUX-THUMB2: ldr r0, [r0]
; LINUX-THUMB2: ldr r0, [r0]

  %0 = call i8* @llvm.frameaddress(i32 3)
  ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
