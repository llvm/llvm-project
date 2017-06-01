; RUN: llc -mtriple=aarch64-apple-darwin -aarch64-cbz-offset-bits=3 < %s | FileCheck %s

; CHECK-LABEL: _split_block_no_fallthrough:
; CHECK: cmn x{{[0-9]+}}, #5
; CHECK-NEXT: b.le [[B2:LBB[0-9]+_[0-9]+]]

; CHECK-NEXT: ; BB#1: ; %b3
; CHECK: ldr [[LOAD:w[0-9]+]]
; CHECK: cbz [[LOAD]], [[SKIP_LONG_B:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: b [[B8:LBB[0-9]+_[0-9]+]]

; CHECK-NEXT: [[SKIP_LONG_B]]:
; CHECK-NEXT: b [[B7:LBB[0-9]+_[0-9]+]]

; CHECK-NEXT: [[B2]]: ; %b2
; CHECK: mov w{{[0-9]+}}, #93
; CHECK: bl _extfunc
; CHECK: cbz w{{[0-9]+}}, [[B7]]

; CHECK-NEXT: [[B8]]: ; %b8
; CHECK-NEXT: ret

; CHECK-NEXT: [[B7]]: ; %b7
; CHECK: mov w{{[0-9]+}}, #13
; CHECK: b _extfunc
define void @split_block_no_fallthrough(i64 %val) #0 {
bb:
  %c0 = icmp sgt i64 %val, -5
  br i1 %c0, label %b3, label %b2

b2:
  %v0 = tail call i32 @extfunc(i32 93)
  %c1 = icmp eq i32 %v0, 0
  br i1 %c1, label %b7, label %b8

b3:
  %v1 = load volatile i32, i32* undef, align 4
  %c2 = icmp eq i32 %v1, 0
  br i1 %c2, label %b7, label %b8

b7:
  %tmp1 = tail call i32 @extfunc(i32 13)
  ret void

b8:
  ret void
}

declare i32 @extfunc(i32) #0

attributes #0 = { nounwind }
