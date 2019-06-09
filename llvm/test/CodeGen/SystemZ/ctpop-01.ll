; Test population-count instruction
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i32 @llvm.ctpop.i32(i32 %a)
declare i64 @llvm.ctpop.i64(i64 %a)

define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: popcnt  %r0, %r2
; CHECK: sllk    %r1, %r0, 16
; CHECK: ar      %r0, %r1
; CHECK: sllk    %r1, %r0, 8
; CHECK: ar      %r0, %r1
; CHECK: srlk    %r2, %r0, 24
; CHECK: br      %r14

  %popcnt = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %popcnt
}

define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: llhr    %r0, %r2
; CHECK: popcnt  %r0, %r0
; CHECK: risblg  %r1, %r0, 16, 151, 8
; CHECK: ar      %r0, %r1
; CHECK: srlk    %r2, %r0, 8
; CHECK: br      %r14
  %and = and i32 %a, 65535
  %popcnt = call i32 @llvm.ctpop.i32(i32 %and)
  ret i32 %popcnt
}

define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: llcr    %r0, %r2
; CHECK: popcnt  %r2, %r0
; CHECK: br      %r14
  %and = and i32 %a, 255
  %popcnt = call i32 @llvm.ctpop.i32(i32 %and)
  ret i32 %popcnt
}

define i64 @f4(i64 %a) {
; CHECK-LABEL: f4:
; CHECK: popcnt  %r0, %r2
; CHECK: sllg    %r1, %r0, 32
; CHECK: agr     %r0, %r1
; CHECK: sllg    %r1, %r0, 16
; CHECK: agr     %r0, %r1
; CHECK: sllg    %r1, %r0, 8
; CHECK: agr     %r0, %r1
; CHECK: srlg    %r2, %r0, 56
; CHECK: br      %r14
  %popcnt = call i64 @llvm.ctpop.i64(i64 %a)
  ret i64 %popcnt
}

define i64 @f5(i64 %a) {
; CHECK-LABEL: f5:
; CHECK: llgfr   %r0, %r2
; CHECK: popcnt  %r0, %r0
; CHECK: sllg    %r1, %r0, 16
; CHECK: algfr   %r0, %r1
; CHECK: sllg    %r1, %r0, 8
; CHECK: algfr   %r0, %r1
; CHECK: srlg    %r2, %r0, 24
  %and = and i64 %a, 4294967295
  %popcnt = call i64 @llvm.ctpop.i64(i64 %and)
  ret i64 %popcnt
}

define i64 @f6(i64 %a) {
; CHECK-LABEL: f6:
; CHECK: llghr   %r0, %r2
; CHECK: popcnt  %r0, %r0
; CHECK: risbg   %r1, %r0, 48, 183, 8
; CHECK: agr     %r0, %r1
; CHECK: srlg    %r2, %r0, 8
; CHECK: br      %r14
  %and = and i64 %a, 65535
  %popcnt = call i64 @llvm.ctpop.i64(i64 %and)
  ret i64 %popcnt
}

define i64 @f7(i64 %a) {
; CHECK-LABEL: f7:
; CHECK: llgcr   %r0, %r2
; CHECK: popcnt  %r2, %r0
; CHECK: br      %r14
  %and = and i64 %a, 255
  %popcnt = call i64 @llvm.ctpop.i64(i64 %and)
  ret i64 %popcnt
}

