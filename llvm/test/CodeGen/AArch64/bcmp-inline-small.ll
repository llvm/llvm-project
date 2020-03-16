; RUN: llc -O2 < %s -mtriple=aarch64-linux-gnu                     | FileCheck %s --check-prefixes=CHECK,CHECKN
; RUN: llc -O2 < %s -mtriple=aarch64-linux-gnu -mattr=strict-align | FileCheck %s --check-prefixes=CHECK,CHECKS

declare i32 @bcmp(i8*, i8*, i64) nounwind readonly
declare i32 @memcmp(i8*, i8*, i64) nounwind readonly

define i1 @test_b2(i8* %s1, i8* %s2) {
entry:
  %bcmp = call i32 @bcmp(i8* %s1, i8* %s2, i64 15)
  %ret = icmp eq i32 %bcmp, 0
  ret i1 %ret

; CHECK-LABEL: test_b2:
; CHECK-NOT:   bl bcmp
; CHECKN:      ldr  x
; CHECKN-NEXT: ldr  x
; CHECKN-NEXT: ldur x
; CHECKN-NEXT: ldur x
; CHECKS-COUNT-30: ldrb w
}

define i1 @test_b2_align8(i8* align 8 %s1, i8* align 8 %s2) {
entry:
  %bcmp = call i32 @bcmp(i8* %s1, i8* %s2, i64 15)
  %ret = icmp eq i32 %bcmp, 0
  ret i1 %ret

; CHECK-LABEL: test_b2_align8:
; CHECK-NOT:   bl bcmp
; CHECKN:      ldr  x
; CHECKN-NEXT: ldr  x
; CHECKN-NEXT: ldur x
; CHECKN-NEXT: ldur x
; CHECKS:      ldr  x
; CHECKS-NEXT: ldr  x
; CHECKS-NEXT: ldr  w
; CHECKS-NEXT: ldr  w
; CHECKS-NEXT: ldrh  w
; CHECKS-NEXT: ldrh  w
; CHECKS-NEXT: ldrb  w
; CHECKS-NEXT: ldrb  w
}

define i1 @test_bs(i8* %s1, i8* %s2) optsize {
entry:
  %memcmp = call i32 @memcmp(i8* %s1, i8* %s2, i64 31)
  %ret = icmp eq i32 %memcmp, 0
  ret i1 %ret

; CHECK-LABEL: test_bs:
; CHECKN-NOT:  bl memcmp
; CHECKN:      ldp  x
; CHECKN-NEXT: ldp  x
; CHECKN-NEXT: ldr  x
; CHECKN-NEXT: ldr  x
; CHECKN-NEXT: ldur x
; CHECKN-NEXT: ldur x
; CHECKS:      bl memcmp
}
