; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=DARWIN-ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=LINUX-ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=DARWIN-THUMB2
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=thumbv7-linux-gnueabi | FileCheck %s --check-prefix=LINUX-THUMB2

define ptr @frameaddr_index0() nounwind {
entry:
; DARWIN-ARM-LABEL: frameaddr_index0:
; DARWIN-ARM: push {r7, lr}
; DARWIN-ARM: mov r7, sp
; DARWIN-ARM: mov r0, r7

; DARWIN-THUMB2-LABEL: frameaddr_index0:
; DARWIN-THUMB2: push {r7, lr}
; DARWIN-THUMB2: mov r7, sp
; DARWIN-THUMB2: mov r0, r7

; LINUX-ARM-LABEL: frameaddr_index0:
; LINUX-ARM: push {r11, lr}
; LINUX-ARM: mov r11, sp
; LINUX-ARM: mov r0, r11

; LINUX-THUMB2-LABEL: frameaddr_index0:
; LINUX-THUMB2: push {r7, lr}
; LINUX-THUMB2: mov r7, sp
; LINUX-THUMB2: mov r0, r7

  %0 = call ptr @llvm.frameaddress(i32 0)
  ret ptr %0
}

define ptr @frameaddr_index1() nounwind {
entry:
; DARWIN-ARM-LABEL: frameaddr_index1:
; DARWIN-ARM: push {r7, lr}
; DARWIN-ARM: mov r7, sp
; DARWIN-ARM: ldr r0, [r7]

; DARWIN-THUMB2-LABEL: frameaddr_index1:
; DARWIN-THUMB2: push {r7, lr}
; DARWIN-THUMB2: mov r7, sp
; DARWIN-THUMB2: ldr r0, [r7]

; LINUX-ARM-LABEL: frameaddr_index1:
; LINUX-ARM: push {r11, lr}
; LINUX-ARM: mov r11, sp
; LINUX-ARM: ldr r0, [r11]

; LINUX-THUMB2-LABEL: frameaddr_index1:
; LINUX-THUMB2: push {r7, lr}
; LINUX-THUMB2: mov r7, sp
; LINUX-THUMB2: mov r0, r7
; LINUX-THUMB2: ldr r0, [r0]

  %0 = call ptr @llvm.frameaddress(i32 1)
  ret ptr %0
}

define ptr @frameaddr_index3() nounwind {
entry:
; DARWIN-ARM-LABEL: frameaddr_index3:
; DARWIN-ARM: push {r7, lr}
; DARWIN-ARM: mov r7, sp
; DARWIN-ARM: ldr r0, [r7]
; DARWIN-ARM: ldr r0, [r0]
; DARWIN-ARM: ldr r0, [r0]

; DARWIN-THUMB2-LABEL: frameaddr_index3:
; DARWIN-THUMB2: push {r7, lr}
; DARWIN-THUMB2: mov r7, sp
; DARWIN-THUMB2: ldr r0, [r7]
; DARWIN-THUMB2: ldr r0, [r0]
; DARWIN-THUMB2: ldr r0, [r0]

; LINUX-ARM-LABEL: frameaddr_index3:
; LINUX-ARM: push {r11, lr}
; LINUX-ARM: mov r11, sp
; LINUX-ARM: ldr r0, [r11]
; LINUX-ARM: ldr r0, [r0]
; LINUX-ARM: ldr r0, [r0]

; LINUX-THUMB2-LABEL: frameaddr_index3:
; LINUX-THUMB2: push {r7, lr}
; LINUX-THUMB2: mov r7, sp
; LINUX-THUMB2: mov r0, r7
; LINUX-THUMB2: ldr r0, [r0]
; LINUX-THUMB2: ldr r0, [r0]
; LINUX-THUMB2: ldr r0, [r0]

  %0 = call ptr @llvm.frameaddress(i32 3)
  ret ptr %0
}

declare ptr @llvm.frameaddress(i32) nounwind readnone
