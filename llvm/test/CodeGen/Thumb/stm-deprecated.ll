; RUN: llc -mtriple=thumbv6m-eabi -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv5e-linux-gnueabi -verify-machineinstrs %s -o - | FileCheck %s

%0 = type { %0*, %0*, i32 }

@x1 = external global %0, align 4
@x2 = external global %0, align 4

; CHECK: str r0, [r1]
; CHECK-NEXT: str r1, [r1, #4]
; CHECK-NOT: stm

define void @foo(i32 %unused, %0* %x) {
  %first = getelementptr inbounds %0, %0* %x, i32 0, i32 0
  %second = getelementptr inbounds %0, %0* %x, i32 0, i32 1
  store %0* @x1, %0** %first
  store %0* %x, %0** %second
  unreachable
}
