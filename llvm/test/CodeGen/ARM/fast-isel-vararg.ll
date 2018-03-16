; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

define i32 @VarArg() nounwind {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %m = alloca i32, align 4
  %n = alloca i32, align 4
  %tmp = alloca i32, align 4
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %j, align 4
  %2 = load i32, i32* %k, align 4
  %3 = load i32, i32* %m, align 4
  %4 = load i32, i32* %n, align 4
; ARM: VarArg
; ARM: mov [[FP:r[0-9]+]], sp
; ARM: sub sp, sp, #{{(36|40)}}
; ARM: ldr r1, {{\[}}[[FP]], #-4]
; ARM: ldr r2, {{\[}}[[FP]], #-8]
; ARM: ldr r3, {{\[}}[[FP]], #-12]
; ARM: ldr [[Ra:r[0-9]+]], {{\[}}[[FP]], #-16]
; ARM: ldr [[Rb:[lr]+[0-9]*]], [sp, #{{(16|20)}}]
; ARM: movw [[Rc:[lr]+[0-9]*]], #5
;   Ra got spilled
; ARM: mov r0, [[Rc]]
; ARM: str {{.*}}, [sp]
; ARM: str [[Rb]], [sp, #4]
; ARM: bl {{_?CallVariadic}}
; THUMB: sub sp, #{{36}}
; THUMB: ldr r1, [sp, #32]
; THUMB: ldr r2, [sp, #28]
; THUMB: ldr r3, [sp, #24]
; THUMB: ldr {{[a-z0-9]+}}, [sp, #20]
; THUMB: ldr.w {{[a-z0-9]+}}, [sp, #16]
; THUMB: str.w {{[a-z0-9]+}}, [sp]
; THUMB: str.w {{[a-z0-9]+}}, [sp, #4]
; THUMB: bl {{_?}}CallVariadic
  %call = call i32 (i32, ...) @CallVariadic(i32 5, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4)
  store i32 %call, i32* %tmp, align 4
  %5 = load i32, i32* %tmp, align 4
  ret i32 %5
}

declare i32 @CallVariadic(i32, ...)
