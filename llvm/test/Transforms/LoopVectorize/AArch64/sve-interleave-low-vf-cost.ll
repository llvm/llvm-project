; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s
; REQUIRES: asserts

target triple = "aarch64"

; Cost-model test for the path where the interleave factor is > the VF.
; The cost is modelled as a contiguous load of the wide vector plus a
; vector.deinterleave4 lowered as a binary tree of (Factor - 1) deinterleave2 shuffles:
;
;   cost = MemCost + (Factor - 1) * LT.first
;  * VF vscale x 2: wide type <vscale x 8 x i16>, LT.first = 1
;       => 1 (load) + 3 * 1 (shuffles) = 4
;  * VF vscale x 4: wide type <vscale x 16 x i16> LT.first = 2
;       => 2 (loads) + 3 * 2 (shuffles) = 8

; CHECK-LABEL: LV: Checking a loop in 'deinterleave4_nxv2i16_load'
; CHECK: Cost of 4 for VF vscale x 2: INTERLEAVE-GROUP with factor 4, ir<%ptr.b>
; CHECK: Cost of 8 for VF vscale x 4: INTERLEAVE-GROUP with factor 4, ir<%ptr.b>
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

attributes #0 = { "target-features"="+sve" }