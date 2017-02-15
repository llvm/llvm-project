; RUN: llc -verify-machineinstrs -o - %s | FileCheck %s
target triple = "arm64--"

; AArch64InstrInfo::optimizeCondBranch() optimizes the
; "x = and y, 256; cmp x, 0; br" from an "and; cbnz" to a tbnz instruction.
; It forgot to clear the a flag resulting in a MachineVerifier complaint.
;
; Writing a stable/simple test is tricky since most tbz instructions are already
; formed in SelectionDAG, optimizeCondBranch() only triggers if the and
; instruction is in a different block than the conditional jump.
;
; CHECK-LABEL: func
; CHECK-NOT: and
; CHECK: tbz
define void @func() {
  %c0 = icmp sgt i64 0, 0
  br i1 %c0, label %b1, label %b6

b1:
  br i1 undef, label %b3, label %b2

b2:
  %v0 = tail call i32 @extfunc()
  br label %b5

b3:
  %v1 = load i32, i32* undef, align 4
  %v2 = and i32 %v1, 256
  br label %b5

b5:
  %v3 = phi i32 [ %v2, %b3 ], [ %v0, %b2 ]
  %c1 = icmp eq i32 %v3, 0
  br i1 %c1, label %b8, label %b7

b6:
  tail call i32 @extfunc()
  ret void

b7:
  tail call i32 @extfunc()
  ret void

b8:
  ret void
}

declare i32 @extfunc()
