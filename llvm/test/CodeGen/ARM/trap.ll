;; Check encodings of trap instructions and that their properties are set
;; correctly so that they are not placed after the stack frame is destroyed.

; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-apple-darwin -trap-func=_trap | FileCheck %s -check-prefix=FUNC
; RUN: llc < %s -mtriple=arm-apple-darwin -trap-func=_trap -O0 | FileCheck %s -check-prefix=FUNC
; RUN: llc < %s -mtriple=armv7 -mattr=+nacl-trap | FileCheck %s -check-prefix=NACL
; RUN: llc < %s -mtriple=armv7 | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7 | FileCheck %s -check-prefix=THUMB

; RUN: llc -mtriple=armv7 -mattr=+nacl-trap -filetype=obj %s -o - \
; RUN:  | llvm-objdump -d --triple=armv7 --mattr=+nacl-trap - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-NACL
; RUN: llc -verify-machineinstrs -fast-isel -mtriple=armv7 -mattr=+nacl-trap -filetype=obj %s -o - \
; RUN:  | llvm-objdump -d --triple=armv7 --mattr=+nacl-trap - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-NACL

; RUN: llc -mtriple=armv7 -filetype=obj %s -o - \
; RUN:  | llvm-objdump -d --triple=armv7 - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-ARM
; RUN: llc -verify-machineinstrs -fast-isel -mtriple=armv7 -filetype=obj %s -o - \
; RUN:  | llvm-objdump -d --triple=armv7 - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-ARM

; RUN: llc -mtriple=thumbv7 -filetype=obj %s -o - \
; RUN:  | llvm-objdump -d --triple=thumbv7 - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-THUMB
; RUN: llc -verify-machineinstrs -fast-isel -mtriple=thumbv7 -filetype=obj %s -o - \
; RUN:  | llvm-objdump -d --triple=thumbv7 - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-THUMB

; rdar://7961298
; rdar://9249183

define void @t() noinline optnone {
entry:
  ;; So that we have a stack frame.
  %1 = alloca i32, align 4
  store volatile i32 0, ptr %1, align 4

; DARWIN-LABEL: t:
; DARWIN:      trap
; DARWIN-NEXT: add sp, sp, #4

; FUNC-LABEL: t:
; FUNC:      bl __trap
; FUNC-NEXT: add sp, sp, #4

; NACL-LABEL: t:
; NACL:      .inst 0xe7fedef0
; NACL-NEXT: add sp, sp, #4

; ARM-LABEL: t:
; ARM:      .inst 0xe7ffdefe
; ARM-NEXT: add sp, sp, #4

; THUMB-LABEL: t:
; THUMB:      .inst.n 0xdefe
; THUMB-NEXT: add sp, #4

; ENCODING-NACL: e7fedef0    trap

; ENCODING-ARM: e7ffdefe    trap

; ENCODING-THUMB: defe  trap

  call void @llvm.trap()
  ret void
}

define void @t2() {
entry:
  ;; So that we have a stack frame.
  %1 = alloca i32, align 4
  store volatile i32 0, ptr %1, align 4

; DARWIN-LABEL: t2:
; DARWIN:      udf #254
; DARWIN-NEXT: add sp, sp, #4

; FUNC-LABEL: t2:
; FUNC:      bl __trap
; FUNC-NEXT: add sp, sp, #4

; NACL-LABEL: t2:
; NACL:      bkpt #0
; NACL-NEXT: add sp, sp, #4

; ARM-LABEL: t2:
; ARM:      bkpt #0
; ARM-NEXT: add sp, sp, #4

; THUMB-LABEL: t2:
; THUMB:      bkpt #0
; THUMB-NEXT: add sp, #4

; ENCODING-NACL: e1200070    bkpt #0

; ENCODING-ARM: e1200070    bkpt #0

; ENCODING-THUMB: be00  bkpt #0

  call void @llvm.debugtrap()
  ret void
}

declare void @llvm.trap() nounwind
declare void @llvm.debugtrap() nounwind
