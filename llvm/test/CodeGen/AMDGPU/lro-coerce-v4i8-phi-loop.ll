; REQUIRES: amdgpu-registered-target
; RUN: opt -S -passes=amdgpu-late-codegenprepare \
; RUN:   -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a %s | FileCheck %s

; Purpose:
;  - Input has a loop-carried PHI of type <4 x i8> and byte-wise adds in the
;    loop header (same basic block as the PHI).
;  - After amdgpu-late-codegenprepare, the PHI must be coerced to i32 across
;    the backedge, and a single dominating "bitcast i32 -> <4 x i8>" must be
;    placed in the header (enabling SDWA-friendly lowering later).
;
; What we check:
;  - PHI is i32 (no loop-carried <4 x i8> PHI remains).
;  - A header-local bitcast i32 -> <4 x i8> exists and feeds the vector add.
;  - The loopexit produces a bitcast <4 x i8> -> i32 for the backedge.

define amdgpu_kernel void @lro_coerce_v4i8_phi(ptr nocapture %p, i32 %n) {
entry:
  br label %loop

loop:
  ; Loop index
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  ; Loop-carried accumulator in vector-of-bytes form (problematic on input).
  %acc = phi <4 x i8> [ zeroinitializer, %entry ], [ %acc.next, %loop ]

  ; Make up four i8 values derived from %i to avoid memory noise.
  %i0 = trunc i32 %i to i8
  %i1i = add i32 %i, 1
  %i1 = trunc i32 %i1i to i8
  %i2i = add i32 %i, 2
  %i2 = trunc i32 %i2i to i8
  %i3i = add i32 %i, 3
  %i3 = trunc i32 %i3i to i8

  ; Pack them into <4 x i8>.
  %v01 = insertelement <4 x i8> zeroinitializer, i8 %i0, i32 0
  %v02 = insertelement <4 x i8> %v01, i8 %i1, i32 1
  %v03 = insertelement <4 x i8> %v02, i8 %i2, i32 2
  %v   = insertelement <4 x i8> %v03, i8 %i3, i32 3

  ; Byte-wise add in the same block as the PHI (this must make coercion profitable).
  %acc.next = add <4 x i8> %acc, %v

  ; Loop control.
  %i.next = add i32 %i, 4
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @lro_coerce_v4i8_phi(
; CHECK: loop:
; CHECK: %i = phi i32
; CHECK-NOT: phi <4 x i8>
; CHECK: %[[ACCI32:[^ ]+]] = phi i32
; CHECK-NEXT: %[[HDRCAST:[^ ]+]] = bitcast i32 %[[ACCI32]] to <4 x i8>
; CHECK: add <4 x i8> %[[HDRCAST]],
; CHECK: br i1

