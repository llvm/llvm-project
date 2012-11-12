; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define zeroext i1 @f1(i8* %x) {
; CHECK: f1:
; CHECK: movb	(%rdi), %al
; CHECK-NEXT: ret

entry:
  %0 = load i8* %x, align 1, !range !0
  %tobool = trunc i8 %0 to i1
  ret i1 %tobool
}

define zeroext i1 @f2(i8* %x) {
; CHECK: f2:
; CHECK: movb	(%rdi), %al
; CHECK-NEXT: ret

entry:
  %0 = load i8* %x, align 1, !range !0
  %tobool = icmp ne i8 %0, 0
  ret i1 %tobool
}

!0 = metadata !{i8 0, i8 2}


; check that we don't build a "trunc" from i1 to i1, which would assert.
define zeroext i1 @f3(i1 %x) {
; CHECK: f3:

entry:
  %tobool = icmp ne i1 %x, 0
  ret i1 %tobool
}

; check that we don't build a trunc when other bits are needed
define zeroext i1 @f4(i32 %x) {
; CHECK: f4:
; CHECK: and

entry:
  %y = and i32 %x, 32768
  %z = icmp ne i32 %y, 0
  ret i1 %z
}
