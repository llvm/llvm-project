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

define void @t() nounwind {
entry:
; DARWIN-LABEL: t:
; DARWIN: trap

; FUNC-LABEL: t:
; FUNC: bl __trap

; NACL-LABEL: t:
; NACL: .inst 0xe7fedef0

; ARM-LABEL: t:
; ARM: .inst 0xe7ffdefe

; THUMB-LABEL: t:
; THUMB: .inst.n 0xdefe

; ENCODING-NACL: f0 de fe e7 trap

; ENCODING-ARM: fe de ff e7 trap

; ENCODING-THUMB: fe de trap

  call void @llvm.trap()
  unreachable
}

define void @t2() nounwind {
entry:
; DARWIN-LABEL: t2:
; DARWIN: udf #254

; FUNC-LABEL: t2:
; FUNC: bl __trap

; NACL-LABEL: t2:
; NACL: bkpt #0

; ARM-LABEL: t2:
; ARM: bkpt #0

; THUMB-LABEL: t2:
; THUMB: bkpt #0

; ENCODING-NACL: 70 00 20 e1 bkpt #0

; ENCODING-ARM: 70 00 20 e1 bkpt #0

; ENCODING-THUMB: 00 be bkpt #0

  call void @llvm.debugtrap()
  unreachable
}

declare void @llvm.trap() nounwind
declare void @llvm.debugtrap() nounwind
