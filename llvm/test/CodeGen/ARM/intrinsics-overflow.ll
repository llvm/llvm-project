; RUN: llc < %s -mtriple=arm-linux -mcpu=generic -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv6m-eabi -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=THUMBV6
; RUN: llc < %s -mtriple=thumbv7-eabi -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=THUMBV7

define i32 @uadd_overflow(i32 %a, i32 %b) #0 {
  %sadd = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %sadd, 1
  %2 = zext i1 %1 to i32
  ret i32 %2

  ; CHECK-LABEL: uadd_overflow:

  ; ARM: adds r[[R0:[0-9]+]], r[[R0]], r[[R1:[0-9]+]]
  ; ARM: mov r[[R2:[0-9]+]], #0
  ; ARM: adc r[[R0]], r[[R2]], #0

  ; THUMBV6: movs    r[[R2:[0-9]+]], #0
  ; THUMBV6: adds    r[[R0:[0-9]+]], r[[R0]], r[[R1:[0-9]+]]
  ; THUMBV6: adcs    r[[R2]], r[[R2]]
  ; THUMBV6: mov     r[[R0]], r[[R2]]

  ; THUMBV7: adds  r[[R0:[0-9]+]], r[[R0]], r[[R1:[0-9]+]]
  ; THUMBV7: mov.w r[[R2:[0-9]+]], #0
  ; THUMBV7: adc   r[[R0]], r[[R2]], #0
}


define i32 @sadd_overflow(i32 %a, i32 %b) #0 {
  %sadd = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %sadd, 1
  %2 = zext i1 %1 to i32
  ret i32 %2

  ; CHECK-LABEL: sadd_overflow:

  ; ARM: adds r[[R2:[0-9]+]], r[[R0:[0-9]+]], r[[R1:[0-9]+]]
  ; ARM: mov r[[R0]], #1
  ; ARM: movvc r[[R0]], #0
  ; ARM: mov pc, lr

  ; THUMBV6: adds    r0, r0, r1
  ; THUMBV6: bvc     .LBB1_2

  ; THUMBV7: adds  r[[R2:[0-9]+]], r[[R0]], r[[R1:[0-9]+]]
  ; THUMBV7: mov.w r[[R0:[0-9]+]], #1
  ; THUMBV7: it    vc
  ; THUMBV7: movvc r[[R0]], #0
}

define i32 @usub_overflow(i32 %a, i32 %b) #0 {
  %sadd = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %sadd, 1
  %2 = zext i1 %1 to i32
  ret i32 %2

  ; CHECK-LABEL: usub_overflow:

  ; ARM: subs    r[[R0:[0-9]+]], r[[R0]], r[[R1:[0-9]+]]
  ; ARM: mov     r[[R2:[0-9]+]], #0
  ; ARM: adc     r[[R0]], r[[R2]], #0
  ; ARM: eor     r[[R0]], r[[R0]], #1

  ; THUMBV6: movs    r[[R2:[0-9]+]], #0
  ; THUMBV6: subs    r[[R0:[0-9]+]], r[[R0]], r[[R1:[0-9]+]]
  ; THUMBV6: adcs    r[[R2]], r[[R2]]
  ; THUMBV6: movs    r[[R0]], #1
  ; THUMBV6: eors    r[[R0]], r[[R2]]

  ; THUMBV7: subs    r[[R0:[0-9]+]], r[[R0]], r[[R1:[0-9]+]]
  ; THUMBV7: mov.w   r[[R2:[0-9]+]], #0
  ; THUMBV7: adc     r[[R0]], r[[R2]], #0
  ; THUMBV7: eor     r[[R0]], r[[R0]], #1

  ; We should know that the overflow is just 1 bit,
  ; no need to clear any other bit
  ; CHECK-NOT: and
}

define i32 @ssub_overflow(i32 %a, i32 %b) #0 {
  %sadd = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b)
  %1 = extractvalue { i32, i1 } %sadd, 1
  %2 = zext i1 %1 to i32
  ret i32 %2

  ; CHECK-LABEL: ssub_overflow:

  ; ARM: mov r[[R2]], #1
  ; ARM: cmp r[[R0]], r[[R1]]
  ; ARM: movvc r[[R2]], #0

  ; THUMBV6: cmp     r0, r1
  ; THUMBV6: bvc     .LBB3_2

  ; THUMBV7: movs  r[[R2:[0-9]+]], #1
  ; THUMBV7: cmp   r[[R0:[0-9]+]], r[[R1:[0-9]+]]
  ; THUMBV7: it    vc
  ; THUMBV7: movvc r[[R2]], #0
  ; THUMBV7: mov   r[[R0]], r[[R2]]
}

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #1
declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #2
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #3
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) #4
