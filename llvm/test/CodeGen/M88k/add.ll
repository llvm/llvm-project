; Test addition/subtraction.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -verify-machineinstrs -m88k-enable-delay-slot-filler=false | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -verify-machineinstrs -m88k-enable-delay-slot-filler=false | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK:       addu %r2, %r2, 512
; CHECK-NEXT:  jmp %r1
  %sum = add i32 %a, 512
  ret i32 %sum
}

define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK:       addu %r2, %r2, 512
; CHECK-NEXT:  jmp %r1
  %sum = add i32 512, %a
  ret i32 %sum
}

define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK:       subu %r2, %r2, 512
; CHECK-NEXT:  jmp %r1
  %sum = sub i32 %a, 512
  ret i32 %sum
}

define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK:       or %r3, %r0, 512
; CHECK-NEXT:  subu %r2, %r3, %r2
; CHECK-NEXT:  jmp %r1
  %sum = sub i32 512, %a
  ret i32 %sum
}

define i64 @f5(i64 %a, i64 %b) {
; CHECK-LABEL: f5:
; CHECK:       addu.co %r3, %r3, %r5
; CHECK-NEXT:  addu.ci %r2, %r2, %r4
; CHECK-NEXT:  jmp %r1
  %sum = add i64 %a, %b
  ret i64 %sum
}

define i64 @f6(i64 %a, i64 %b) {
; CHECK-LABEL: f6:
; CHECK:       subu.co %r3, %r3, %r5
; CHECK-NEXT:  subu.ci %r2, %r2, %r4
; CHECK-NEXT:  jmp %r1
  %sum = sub i64 %a, %b
  ret i64 %sum
}

; Special case: return (a == 0) + b
define i32 @f7(i32 %a, i32 %b) {
; CHECK-LABEL: f7:
; CHECK:       subu.co %r2, %r0, %r2
; CHECK-NEXT:  addu.ci %r2, %r3, %r0
; CHECK-NEXT:  jmp %r1
  %cmp = icmp eq i32 %a, 0
  %conv = zext i1 %cmp to i32
  %sum = add i32 %conv, %b
  ret i32 %sum
}

; Same, but using intrinsic
define i32 @f8(i32 %a, i32 %b) {
; CHECK-LABEL: f8:
; CHECK:       subu.co %r2, %r0, %r2
; CHECK-NEXT:  addu.ci %r2, %r3, %r0
; CHECK-NEXT:  jmp %r1
  %res = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 0, i32 %a)
  %carrybit = extractvalue { i32, i1 } %res, 1
  %carry = zext i1 %carrybit to i32
  %sum = add i32 %carry, %b
  ret i32 %sum
}

declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32)

; Special case: return (a >= b) + c
define i32 @f9(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f9:
; CHECK:       subu.co %r2, %r2, %r3
; CHECK-NEXT:  addu.ci %r2, %r4, %r0
; CHECK-NEXT:  jmp %r1
  %cmp = icmp uge i32 %a, %b
  %conv = zext i1 %cmp to i32
  %sum = add i32 %conv, %c
  ret i32 %sum
}

; Special case: return (a <= b) + c
define i32 @f10(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f10:
; CHECK:       subu.co %r2, %r3, %r2
; CHECK-NEXT:  addu.ci %r2, %r4, %r0
; CHECK-NEXT:  jmp %r1
  %cmp = icmp ule i32 %a, %b
  %conv = zext i1 %cmp to i32
  %sum = add i32 %conv, %c
  ret i32 %sum
}

; Special case: return a - (b < c)
define i32 @f11(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f11:
; CHECK:       subu.co %r3, %r3, %r4
; CHECK-NEXT:  subu.ci %r2, %r2, %r0
; CHECK-NEXT:  jmp %r1
  %cmp = icmp ult i32 %b, %c
  %conv = zext i1 %cmp to i32
  %dif = sub i32 %a, %conv
  ret i32 %dif
}

; Special case: return a - (b > c)
define i32 @f12(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f12:
; CHECK:       subu.co %r3, %r4, %r3
; CHECK-NEXT:  subu.ci %r2, %r2, %r0
; CHECK-NEXT:  jmp %r1
  %cmp = icmp ugt i32 %b, %c
  %conv = zext i1 %cmp to i32
  %dif = sub i32 %a, %conv
  ret i32 %dif
}

; Special case: return a - (b != 0)
define i32 @f13(i32 %a, i32 %b) {
; CHECK-LABEL: f13:
; CHECK:       subu.co %r3, %r0, %r3
; CHECK-NEXT:  subu.ci %r2, %r2, %r0
; CHECK-NEXT:  jmp %r1
  %cmp = icmp ne i32 %b, 0
  %conv = zext i1 %cmp to i32
  %dif = sub i32 %a, %conv
  ret i32 %dif
}

; Special case: return a - (b >= 0)
define i32 @f14(i32 %a, i32 %b) {
; CHECK-LABEL: f14:
; CHECK:       addu.co %r3, %r3, %r3
; CHECK-NEXT:  subu.ci %r2, %r2, %r0
; CHECK-NEXT:  jmp %r1
  %cmp = icmp sge i32 %b, 0
  %conv = zext i1 %cmp to i32
  %dif = sub i32 %a, %conv
  ret i32 %dif
}

; Special case: return a - (0 <= b)
define i32 @f15(i32 %a, i32 %b) {
; CHECK-LABEL: f15:
; CHECK:       addu.co %r3, %r3, %r3
; CHECK-NEXT:  subu.ci %r2, %r2, %r0
; CHECK-NEXT:  jmp %r1
  %cmp = icmp sle i32 0, %b
  %conv = zext i1 %cmp to i32
  %dif = sub i32 %a, %conv
  ret i32 %dif
}
