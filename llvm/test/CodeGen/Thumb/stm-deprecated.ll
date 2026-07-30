; RUN: llc -mtriple=thumbv6m-eabi -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-T1
; RUN: llc -mtriple=thumbv5e-linux-gnueabi -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-T1
; RUN: llc -mtriple=thumbv7m -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-T2
; RUN: llc -mtriple=thumbv7a -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-T2

%struct1 = type { ptr, ptr, i32 }
%struct2 = type { i32, i32, ptr }

@x1 = external global %struct1, align 4
@x2 = external global %struct1, align 4

declare void @fn1(i32, i32)
declare void @fn2(ptr)

; CHECK-LABEL: test1:
; CHECK-T1: str r0, [r1]
; CHECK-T1-NEXT: str r1, [r1, #4]
; CHECK-T2: strd r0, r1, [r1]
; CHECK-NOT: stm
define void @test1(i32 %unused, ptr %x) {
  %second = getelementptr inbounds %struct1, ptr %x, i32 0, i32 1
  store ptr @x1, ptr %x
  store ptr %x, ptr %second
  ret void
}

; CHECK-LABEL: test2:
; CHECK-T1: str r0, [r2]
; CHECK-T1-NEXT: str r1, [r2, #4]
; CHECK-T1-NEXT: str r2, [r2, #8]
; CHECK-T2: stm.w r2, {r0, r1, r2}
; CHECK-NOT: stm r2!, {r0, r1, r2}
define i32 @test2(i32 %a, i32 %b, ptr %p) {
entry:
  %p2 = getelementptr inbounds %struct2, ptr %p, i32 0, i32 1
  %p3 = getelementptr inbounds %struct2, ptr %p, i32 0, i32 2
  store i32 %a, ptr %p, align 4
  store i32 %b, ptr %p2, align 4
  store ptr %p, ptr %p3, align 4
  call void @fn1(i32 %a, i32 %b)
  ret i32 0
}

; CHECK-LABEL: test3:
; CHECK-T1: str r0, [r2]
; CHECK-T1-NEXT: str r1, [r2, #4]
; CHECK-T1-NEXT: str r2, [r2, #8]
; CHECK-T2: stm.w r2, {r0, r1, r2}
; CHECK-NOT: stm r2!, {r0, r1, r2}
define i32 @test3(i32 %a, i32 %b, ptr %p) {
entry:
  %p2 = getelementptr inbounds %struct2, ptr %p, i32 0, i32 1
  %p3 = getelementptr inbounds %struct2, ptr %p, i32 0, i32 2
  store i32 %a, ptr %p, align 4
  store i32 %b, ptr %p2, align 4
  store ptr %p, ptr %p3, align 4
  %p4 = getelementptr inbounds %struct2, ptr %p, i32 1
  call void @fn2(ptr %p4)
  ret i32 0
}

; FIXME: We should be using stm in both thumb1 and thumb2
; CHECK-LABEL: test4:
; CHECK-T1: str r0, [r0]
; CHECK-T1-NEXT: str r1, [r0, #4]
; CHECK-T1-NEXT: str r2, [r0, #8]
; CHECK-T2: stm r0!, {r0, r1, r2}
define i32 @test4(ptr %p, ptr %q, i32 %a) {
entry:
  %p2 = getelementptr inbounds %struct1, ptr %p, i32 0, i32 1
  %p3 = getelementptr inbounds %struct1, ptr %p, i32 0, i32 2
  store ptr %p, ptr %p, align 4
  store ptr %q, ptr %p2, align 4
  store i32 %a, ptr %p3, align 4
  call void @fn1(i32 %a, i32 %a)
  ret i32 0
}
