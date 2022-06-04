; Test LD instructions.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 | FileCheck %s

@mem32 = common global i32 0, align 4
@mem16 = common global i16 0, align 2
@mem8 = common global i8 0, align 1

define i32 @get_mem32() {
; CHECK-LABEL: get_mem32:
; CHECK: or.u %r2, %r0, %hi16(mem32)
; CHECK-NEXT: jmp.n %r1
; CHECK-NEXT: ld %r2, %r2, %lo16(mem32)
  %res = load i32, ptr @mem32, align 4
  ret i32 %res
}

define i32 @get_mem16s() {
; CHECK-LABEL: get_mem16s:
; CHECK: or.u %r2, %r0, %hi16(mem16)
; CHECK-NEXT: jmp.n %r1
; CHECK-NEXT: ld.h %r2, %r2, %lo16(mem16)
  %val = load i16, ptr @mem16, align 2
  %res = sext i16 %val to i32
  ret i32 %res
}

define i32 @get_mem16u() {
; CHECK-LABEL: get_mem16u:
; CHECK: or.u %r2, %r0, %hi16(mem16)
; CHECK-NEXT: jmp.n %r1
; CHECK-NEXT: ld.hu %r2, %r2, %lo16(mem16)
  %val = load i16, ptr @mem16, align 2
  %res = zext i16 %val to i32
  ret i32 %res
}

define i32 @get_mem8s() {
; CHECK-LABEL: get_mem8s:
; CHECK: or.u %r2, %r0, %hi16(mem8)
; CHECK-NEXT: jmp.n %r1
; CHECK-NEXT: ld.b %r2, %r2, %lo16(mem8)
  %val = load i8, ptr @mem8, align 2
  %res = sext i8 %val to i32
  ret i32 %res
}

define i32 @get_mem8u() {
; CHECK-LABEL: get_mem8u:
; CHECK: or.u %r2, %r0, %hi16(mem8)
; CHECK-NEXT: jmp.n %r1
; CHECK-NEXT: ld.bu %r2, %r2, %lo16(mem8)
  %val = load i8, ptr @mem8, align 2
  %res = zext i8 %val to i32
  ret i32 %res
}

; CHECK: .comm   mem32,4,4
; CHECK: .comm   mem16,2,2
; CHECK: .comm   mem8,1,1