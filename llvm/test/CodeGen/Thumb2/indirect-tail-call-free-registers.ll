; RUN: llc %s -o - -mtriple=thumbv8m.main -mattr=+vfp4 | FileCheck %s

;; No outgoing arguments, plenty of free registers to hold the target address.
define void @test0(ptr %fptr) {
; CHECK-LABEL: test0:
; CHECK: bx {{r0|r1|r2|r3|r12}}
entry:
  tail call void %fptr()
  ret void
}

;; Four integer outgoing arguments, which use up r0-r3.
define void @test1(ptr %fptr) {
; CHECK-LABEL: test1:
; CHECK: bx r12
entry:
  tail call void %fptr(i32 0, i32 0, i32 0, i32 0)
  ret void
}

;; Four integer outgoing arguments, which use up r0-r3, and sign-return-address
;; uses r12, so we can never tail-call this.
define void @test2(ptr %fptr) "sign-return-address"="all" {
; CHECK-LABEL: test2:
; CHECK: blx
  entry:
  tail call void %fptr(i32 0, i32 0, i32 0, i32 0)
  ret void
}

;; An i32 and an i64 argument, which uses r0, r2 and r3 for arguments, leaving
;; r1 free for the address.
define void @test3(ptr %fptr) {
; CHECK-LABEL: test3:
; CHECK: bx {{r1|r12}}
entry:
  tail call void %fptr(i32 0, i64 0)
  ret void
}

;; Four float arguments, using the soft-float calling convention, which uses
;; r0-r3.
define void @test4(ptr %fptr) {
; CHECK-LABEL: test4:
; CHECK: bx r12
entry:
  tail call arm_aapcscc void %fptr(float 0.0, float 0.0, float 0.0, float 0.0)
  ret void
}

;; Four float arguments, using the soft-float calling convention, which uses
;; r0-r3, and sign-return-address uses r12. Currently fails with "ran out of
;; registers during register allocation".
define void @test5(ptr %fptr) "sign-return-address"="all" {
; CHECK-LABEL: test5:
; CHECK: blx
entry:
  tail call arm_aapcscc void %fptr(float 0.0, float 0.0, float 0.0, float 0.0)
  ret void
}

;; Four float arguments, using the hard-float calling convention, which uses
;; s0-s3, leaving the all of the integer registers free for the address.
define void @test6(ptr %fptr) {
; CHECK-LABEL: test6:
; CHECK: bx {{r0|r1|r2|r3|r12}}
entry:
  tail call arm_aapcs_vfpcc void %fptr(float 0.0, float 0.0, float 0.0, float 0.0)
  ret void
}

;; Four float arguments, using the hard-float calling convention, which uses
;; s0-s3, leaving r0-r3 free for the address, with r12 used for
;; sign-return-address.
define void @test7(ptr %fptr) "sign-return-address"="all" {
; CHECK-LABEL: test7:
; CHECK: bx {{r0|r1|r2|r3}}
entry:
  tail call arm_aapcs_vfpcc void %fptr(float 0.0, float 0.0, float 0.0, float 0.0)
  ret void
}

;; Two double arguments, using the soft-float calling convention, which uses
;; r0-r3.
define void @test8(ptr %fptr) {
; CHECK-LABEL: test8:
; CHECK: bx r12
entry:
  tail call arm_aapcscc void %fptr(double 0.0, double 0.0)
  ret void
}

;; Two double arguments, using the soft-float calling convention, which uses
;; r0-r3, and sign-return-address uses r12, so we can't tail-call this.
define void @test9(ptr %fptr) "sign-return-address"="all" {
; CHECK-LABEL: test9:
; CHECK: blx
entry:
  tail call arm_aapcscc void %fptr(double 0.0, double 0.0)
  ret void
}

;; Four integer arguments (one on the stack), but dut to alignment r1 is left
;; empty, so can be used for the tail-call.
define void @test10(ptr %fptr, i64 %b, i32 %c) "sign-return-address"="all" {
; CHECK-LABEL: test10:
; CHECK: bx r1
entry:
  tail call void %fptr(i32 0, i64 %b, i32 %c)
  ret void
}
