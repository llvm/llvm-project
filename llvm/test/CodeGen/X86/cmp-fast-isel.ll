; RUN: llc -mtriple=x86_64-linux -fast-isel -show-mc-encoding < %s | FileCheck %s

; pr22854

define i32 @f1(i16 %x) {
; CHECK-LABEL: f1:
; CHECK: cmpw	$42, %di               # encoding: [0x66,0x83,0xff,0x2a]
bb0:
  %cmp = icmp ne i16 %x, 42
  br i1 %cmp, label %bb3, label %bb7

bb3:
  ret i32 1

bb7:
  ret i32 2
}

define i32 @f2(i32 %x) {
; CHECK-LABEL: f2:
; CHECK: cmpl	$42, %edi               # encoding: [0x83,0xff,0x2a]
bb0:
  %cmp = icmp ne i32 %x, 42
  br i1 %cmp, label %bb3, label %bb7

bb3:
  ret i32 1

bb7:
  ret i32 2
}

define i32 @f3(i64 %x) {
; CHECK-LABEL: f3:
; CHECK: cmpq	$42, %rdi               # encoding: [0x48,0x83,0xff,0x2a]
bb0:
  %cmp = icmp ne i64 %x, 42
  br i1 %cmp, label %bb3, label %bb7

bb3:
  ret i32 1

bb7:
  ret i32 2
}
