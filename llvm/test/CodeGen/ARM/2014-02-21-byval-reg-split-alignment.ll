; RUN: llc -mtriple=arm-linux-gnueabihf < %s | FileCheck %s

%struct4bytes = type { i32 }
%struct8bytes8align = type { i64 }
%struct12bytes = type { i32, i32, i32 }

declare void @useIntPtr(%struct4bytes*)
declare void @useLong(i64)
declare void @usePtr(%struct8bytes8align*)

; a -> r0
; b -> r1..r3
; c -> sp+0..sp+7
define void @foo1(i32 %a, %struct12bytes* byval %b, i64 %c) {
; CHECK-LABEL: foo1
; CHECK: sub  sp, sp, #12
; CHECK: push  {r11, lr}
; CHECK: sub sp, sp, #4
; CHECK: add  [[SCRATCH:r[0-9]+]], sp, #12
; CHECK: stm  [[SCRATCH]], {r1, r2, r3}
; CHECK: ldr  r0, [sp, #24]
; CHECK: ldr  r1, [sp, #28]
; CHECK: bl  useLong
; CHECK: pop  {r11, lr}
; CHECK: add  sp, sp, #12

  call void @useLong(i64 %c)
  ret void
}

; a -> r0
; b -> r2..r3
define void @foo2(i32 %a, %struct8bytes8align* byval %b) {
; CHECK-LABEL: foo2
; CHECK: sub  sp, sp, #8
; CHECK: push  {r11, lr}
; CHECK: add  r0, sp, #8
; CHECK-DAG: str  r3, [sp, #12]
; CHECK-DAG: str  r2, [sp, #8]
; CHECK: bl   usePtr
; CHECK: pop  {r11, lr}
; CHECK: add  sp, sp, #8

  call void @usePtr(%struct8bytes8align* %b)
  ret void
}

; a -> r0..r1
; b -> r2
define void @foo3(%struct8bytes8align* byval %a, %struct4bytes* byval %b) {
; CHECK-LABEL: foo3
; CHECK: sub  sp, sp, #16
; CHECK: push  {r11, lr}
; CHECK: add  [[SCRATCH:r[0-9]+]], sp, #8
; CHECK: stm  [[SCRATCH]], {r0, r1, r2}
; CHECK: add  r0, sp, #8
; CHECK: bl   usePtr
; CHECK: pop  {r11, lr}
; CHECK: add  sp, sp, #16

  call void @usePtr(%struct8bytes8align* %a)
  ret void
}

; a -> r0
; b -> r2..r3
define void @foo4(%struct4bytes* byval %a, %struct8bytes8align* byval %b) {
; CHECK-LABEL: foo4
; CHECK: sub     sp, sp, #16
; CHECK: push    {r11, lr}
; CHECK: str     r0, [sp, #8]
; CHECK: add     r0, sp, #16
; CHECK-DAG: str     r3, [sp, #20]
; CHECK-DAG: str     r2, [sp, #16]
; CHECK: bl      usePtr
; CHECK: pop     {r11, lr}
; CHECK: add     sp, sp, #16
; CHECK: mov     pc, lr

  call void @usePtr(%struct8bytes8align* %b)
  ret void
}

; a -> r0..r1
; b -> r2
; c -> r3
define void @foo5(%struct8bytes8align* byval %a, %struct4bytes* byval %b, %struct4bytes* byval %c) {
; CHECK-LABEL: foo5
; CHECK: sub     sp, sp, #16
; CHECK: push    {r11, lr}
; CHECK: add     [[SCRATCH:r[0-9]+]], sp, #8
; CHECK: stm     [[SCRATCH]], {r0, r1, r2, r3}
; CHECK: add     r0, sp, #8
; CHECK: bl      usePtr
; CHECK: pop     {r11, lr}
; CHECK: add     sp, sp, #16
; CHECK: mov     pc, lr

  call void @usePtr(%struct8bytes8align* %a)
  ret void
}

; a..c -> r0..r2
; d -> sp+0..sp+7
define void @foo6(i32 %a, i32 %b, i32 %c, %struct8bytes8align* byval %d) {
; CHECK-LABEL: foo6
; CHECK: push {r11, lr}
; CHECK: add  r0, sp, #8
; CHECK: bl   usePtr
; CHECK: pop  {r11, lr}
; CHECK: mov  pc, lr

  call void @usePtr(%struct8bytes8align* %d)
  ret void
}
