; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target triple = "aarch64"

; Cost-model test for the path where the interleave factor is > the VF.
; The cost is modelled as a contiguous load of the wide vector plus a
; vector.deinterleave4 lowered as a binary tree of Log2(Factor) levels,
; each with Factor operations.
;
;   cost = MemCost + (Factor * LegalizationCost) + (Factor * Log2(Factor))
;  * VF vscale x 2: wide type <vscale x 8 x i16>, MemCost = 1, Subvector Legalization Cost = 1
;       => 1 + (4 * 1) + (4 * 2) = 13
;  * VF vscale x 4: wide type <vscale x 16 x i16>, MemCost = 2, Subvector Legalization Cost = 1
;       => 2 + (4 * 1) + (4 * 2) = 14

; CHECK-LABEL: LV: Checking a loop in 'deinterleave4_nxv2i16_load'
; CHECK: Cost of 13 for VF vscale x 2: INTERLEAVE-GROUP with factor 4, ir<%ptr.b>
; CHECK: Cost of 14 for VF vscale x 4: INTERLEAVE-GROUP with factor 4, ir<%ptr.b>
; CHECK: LV: Selecting VF: vscale x 2
define void @deinterleave4_nxv2i16_load(ptr noalias readonly %src, ptr noalias %out, i64 %n) #0 {
entry:
  br label %loop

loop:
  %iv    = phi i64   [ 0,   %entry ], [ %iv.next, %loop ]
  %sum.b = phi double[ 0.0, %entry ], [ %add.b,   %loop ]
  %sum.g = phi double[ 0.0, %entry ], [ %add.g,   %loop ]
  %sum.r = phi double[ 0.0, %entry ], [ %add.r,   %loop ]
  %sum.a = phi double[ 0.0, %entry ], [ %add.a,   %loop ]

  %base  = shl nuw i64 %iv, 2
  %ptr.b = getelementptr inbounds i16, ptr %src, i64 %base
  %load.b = load i16, ptr %ptr.b, align 2

  %off.g = add nuw i64 %base, 1
  %ptr.g = getelementptr inbounds i16, ptr %src, i64 %off.g
  %load.g = load i16, ptr %ptr.g, align 2

  %off.r = add nuw i64 %base, 2
  %ptr.r = getelementptr inbounds i16, ptr %src, i64 %off.r
  %load.r = load i16, ptr %ptr.r, align 2

  %off.a = add nuw i64 %base, 3
  %ptr.a = getelementptr inbounds i16, ptr %src, i64 %off.a
  %load.a = load i16, ptr %ptr.a, align 2

  %ext.b = uitofp i16 %load.b to double
  %ext.g = uitofp i16 %load.g to double
  %ext.r = uitofp i16 %load.r to double
  %ext.a = uitofp i16 %load.a to double

  %add.b = fadd double %sum.b, %ext.b
  %add.g = fadd double %sum.g, %ext.g
  %add.r = fadd double %sum.r, %ext.r
  %add.a = fadd double %sum.a, %ext.a

  %iv.next = add nuw nsw i64 %iv, 1
  %done    = icmp eq i64 %iv.next, %n
  br i1 %done, label %exit, label %loop

exit:
  store double %add.b, ptr %out, align 8
  %out1 = getelementptr inbounds double, ptr %out, i64 1
  store double %add.g, ptr %out1, align 8
  %out2 = getelementptr inbounds double, ptr %out, i64 2
  store double %add.r, ptr %out2, align 8
  %out3 = getelementptr inbounds double, ptr %out, i64 3
  store double %add.a, ptr %out3, align 8
  ret void
}

; Check that the increased low-VF interleaved-store cost prevents selection of
; an SVE epilogue.

; For VF vscale x 4:
;   load cost  = 2 + (4 * 1) + (4 * 2) = 14
;   store cost = 1 + (4 * 4) + (4 * 2) = 25
;
; This makes the fixed VF 8 epilogue preferable to VF vscale x 4.
;
; CHECK-LABEL: LV: Checking a loop in 'deinterleave4_nxv4i16_load_interleave4_nxv4i8_store'
; CHECK: Cost of 14 for VF vscale x 4: INTERLEAVE-GROUP with factor 4
; CHECK: Cost of 25 for VF vscale x 4: INTERLEAVE-GROUP with factor 4
; CHECK: LV: Selecting VF: vscale x 16
; CHECK: LEV: Vectorizing epilogue loop with VF = 8
define void @deinterleave4_nxv4i16_load_interleave4_nxv4i8_store(
    ptr readonly %src, ptr writeonly %out, i32 %n) #0 {
entry:
  %empty = icmp eq i32 %n, 0
  br i1 %empty, label %exit, label %loop

loop:
  %src.iv = phi ptr [ %src.next, %loop ], [ %src, %entry ]
  %out.iv = phi ptr [ %out.next, %loop ], [ %out, %entry ]
  %iv = phi i32 [ %iv.next, %loop ], [ %n, %entry ]

  %ptr.g = getelementptr inbounds i16, ptr %src.iv, i64 1
  %ptr.r = getelementptr inbounds i16, ptr %src.iv, i64 2
  %ptr.a = getelementptr inbounds i16, ptr %src.iv, i64 3
  %load.b = load i16, ptr %src.iv, align 2
  %load.g = load i16, ptr %ptr.g, align 2
  %load.r = load i16, ptr %ptr.r, align 2
  %load.a = load i16, ptr %ptr.a, align 2

  %shift.b = lshr i16 %load.b, 8
  %shift.g = lshr i16 %load.g, 8
  %shift.r = lshr i16 %load.r, 8
  %shift.a = lshr i16 %load.a, 8
  %trunc.b = trunc nuw i16 %shift.b to i8
  %trunc.g = trunc nuw i16 %shift.g to i8
  %trunc.r = trunc nuw i16 %shift.r to i8
  %trunc.a = trunc nuw i16 %shift.a to i8

  %out.g = getelementptr inbounds i8, ptr %out.iv, i64 1
  %out.r = getelementptr inbounds i8, ptr %out.iv, i64 2
  %out.a = getelementptr inbounds i8, ptr %out.iv, i64 3
  store i8 %trunc.b, ptr %out.iv, align 1
  store i8 %trunc.g, ptr %out.g, align 1
  store i8 %trunc.r, ptr %out.r, align 1
  store i8 %trunc.a, ptr %out.a, align 1

  %src.next = getelementptr inbounds i16, ptr %src.iv, i64 4
  %out.next = getelementptr inbounds i8, ptr %out.iv, i64 4
  %iv.next = add nsw i32 %iv, -1
  %done = icmp eq i32 %iv.next, 0
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { "target-features"="+sve" }
