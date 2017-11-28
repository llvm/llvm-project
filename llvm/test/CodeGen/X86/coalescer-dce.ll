; RUN: llc < %s -verify-machineinstrs -disable-fp-elim -disable-machine-dce -verify-coalescing
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

; This test case has a sub-register join followed by a remat:
;
; 256L    %vreg2<def> = COPY %vreg7:sub_32bit<kill>; GR32:%vreg2 GR64:%vreg7
;         Considering merging %vreg2 with %vreg7:sub_32bit
;         Cross-class to GR64.
;                 RHS = %vreg2 = [256d,272d:0)  0@256d
;                 LHS = %vreg7 = [208d,256d:0)[304L,480L:0)  0@208d
;                 updated: 272L   %vreg0<def> = COPY %vreg7:sub_32bit<kill>; GR32:%vreg0 GR64:%vreg7
;         Joined. Result = %vreg7 = [208d,272d:0)[304L,480L:0)  0@208d
;
; 272L    %vreg10:sub_32bit<def> = COPY %vreg7:sub_32bit<kill>, %vreg10<imp-def>; GR64:%vreg10,%vreg7
;         Considering merging %vreg7 with %vreg10
;                 RHS = %vreg7 = [208d,272d:0)[304L,480L:0)  0@208d
;                 LHS = %vreg10 = [16d,64L:2)[64L,160L:1)[192L,240L:1)[272d,304L:3)[304L,352d:1)[352d,400d:0)[400d,400S:4)  0@352d 1@64L-phidef 2@16d-phikill 3@272d-phikill 4@400d
; Remat: %vreg10<def> = MOV64r0 %vreg10<imp-def>, %eflags<imp-def,dead>, %vreg10<imp-def>; GR64:%vreg10
; Shrink: %vreg7 = [208d,272d:0)[304L,480L:0)  0@208d
;  live-in at 240L
;  live-in at 416L
;  live-in at 320L
;  live-in at 304L
; Shrunk: %vreg7 = [208d,256d:0)[304L,480L:0)  0@208d
;
; The COPY at 256L is rewritten as a partial def, and that would artificially
; extend the live range of %vreg7 to end at 256d.  When the joined copy is
; removed, -verify-coalescing complains about the dangling kill.
;
; <rdar://problem/9967101>

define void @f1() nounwind uwtable ssp {
bb:
  br label %bb1

bb1:
  %tmp = phi i32 [ 0, %bb ], [ %tmp21, %bb20 ]
  br label %bb2

bb2:
  br i1 undef, label %bb5, label %bb8

bb4:
  br i1 undef, label %bb2, label %bb20

bb5:
  br i1 undef, label %bb4, label %bb20

bb8:
  %tmp9 = phi i32 [ %tmp24, %bb23 ], [ 0, %bb2 ]
  br i1 false, label %bb41, label %bb10

bb10:
  %tmp11 = sub nsw i32 %tmp9, %tmp
  br i1 false, label %bb2, label %bb26

bb20:
  %tmp21 = phi i32 [ undef, %bb4 ], [ undef, %bb5 ], [ %tmp9, %bb27 ], [ undef, %bb32 ]
  %tmp22 = phi i32 [ undef, %bb4 ], [ undef, %bb5 ], [ %tmp11, %bb27 ], [ undef, %bb32 ]
  br label %bb1

bb23:
  %tmp24 = add nsw i32 %tmp9, 1
  br label %bb8

bb26:
  br i1 undef, label %bb27, label %bb32

bb27:
  %tmp28 = zext i32 %tmp11 to i64
  %tmp30 = icmp eq i64 undef, %tmp28
  br i1 %tmp30, label %bb20, label %bb27

bb32:
  br i1 undef, label %bb20, label %bb23

bb41:
  ret void
}
