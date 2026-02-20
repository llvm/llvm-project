; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -vectorizer-maximize-bandwidth=false -debug-only=loop-vectorize,vplan -disable-output -force-vector-interleave=1 -enable-epilogue-vectorization=false -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOMAX
; RUN: opt -passes=loop-vectorize -vectorizer-maximize-bandwidth=true -debug-only=loop-vectorize,vplan -disable-output -force-vector-interleave=1 -enable-epilogue-vectorization=false -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-REGS-VP
; RUN: opt -passes=loop-vectorize -vectorizer-maximize-bandwidth=true -debug-only=loop-vectorize,vplan -disable-output -force-target-num-vector-regs=1 -force-vector-interleave=1 -enable-epilogue-vectorization=false -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOREGS-VP

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-unknown-elf"

; The use of the dotp instruction means we never have an i32 vector, so we don't
; get any spills normally and with a reduced number of registers the number of
; spills is small enough that it doesn't prevent use of a larger VF.
define i32 @dotp(ptr %a, ptr %b) #0 {
; CHECK-LABEL: LV: Checking a loop in 'dotp'
;
; CHECK-NOMAX: Cost for VF vscale x 4: 6 (Estimated cost per lane: 1.5)
; CHECK-NOMAX: LV: Selecting VF: vscale x 4.
;
; CHECK-REGS-VP: Cost for VF vscale x 4: 6 (Estimated cost per lane: 1.5)
; CHECK-REGS-VP: Cost for VF vscale x 8: 6 (Estimated cost per lane: 0.8)
; CHECK-REGS-VP: Cost for VF vscale x 16: 5 (Estimated cost per lane: 0.3)
; CHECK-REGS-VP: LV: Selecting VF: vscale x 16.
;
; CHECK-NOREGS-VP: Cost for VF vscale x 4: 6 (Estimated cost per lane: 1.5)
; CHECK-NOREGS-VP: LV(REG): Cost of 4 from 2 spills of Generic::VectorRC
; CHECK-NOREGS-VP-NEXT: Cost for VF vscale x 8: 14 (Estimated cost per lane: 1.8)
; CHECK-NOREGS-VP: LV(REG): Cost of 4 from 2 spills of Generic::VectorRC
; CHECK-NOREGS-VP-NEXT: Cost for VF vscale x 16: 13 (Estimated cost per lane: 0.8)
; CHECK-NOREGS-VP: LV: Selecting VF: vscale x 16.
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %accum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %gep.a = getelementptr i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a, align 1
  %ext.a = zext i8 %load.a to i32
  %gep.b = getelementptr i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b, align 1
  %ext.b = zext i8 %load.b to i32
  %mul = mul i32 %ext.b, %ext.a
  %add = add i32 %accum, %mul
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %for.exit, label %for.body

for.exit:                        ; preds = %for.body
  ret i32 %add
}

; The largest type used in the loop is small enough that we already consider all
; VFs and maximize-bandwidth does nothing.
define void @type_too_small(ptr %a, ptr %b) #0 {
; CHECK-LABEL: LV: Checking a loop in 'type_too_small'
; CHECK: Cost for VF vscale x 4: 6 (Estimated cost per lane: 1.5)
; CHECK: Cost for VF vscale x 8: 6 (Estimated cost per lane: 0.8)
; CHECK: Cost for VF vscale x 16: 6 (Estimated cost per lane: 0.4)
; CHECK: LV: Selecting VF: vscale x 16.
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr i8, ptr %a, i64 %iv
  %load.a = load i8, ptr %gep.a, align 1
  %gep.b = getelementptr i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b, align 1
  %add = add i8 %load.a, %load.b
  store i8 %add, ptr %gep.a, align 1
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1024
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; With reduced number of registers the spills from high pressure are enough that
; we use the same VF as if we hadn't maximized the bandwidth.
define void @high_pressure(ptr %a, ptr %b) #0 {
; CHECK-LABEL: LV: Checking a loop in 'high_pressure'
;
; CHECK-NOMAX: Cost for VF vscale x 4: 6 (Estimated cost per lane: 1.5)
; CHECK-NOMAX: LV: Selecting VF: vscale x 4.
;
; CHECK-REGS-VP: Cost for VF vscale x 4: 6 (Estimated cost per lane: 1.5)
; CHECK-REGS-VP: Cost for VF vscale x 8: 10 (Estimated cost per lane: 1.2)
; CHECK-REGS-VP: Cost for VF vscale x 16: 21 (Estimated cost per lane: 1.3)
; CHECK-REGS-VP: LV: Selecting VF: vscale x 8.

; CHECK-NOREGS-VP: Cost for VF vscale x 4: 6 (Estimated cost per lane: 1.5)
; CHECK-NOREGS-VP: LV(REG): Cost of 12 from 3 spills of Generic::VectorRC
; CHECK-NOREGS-VP-NEXT: Cost for VF vscale x 8: 26 (Estimated cost per lane: 3.2)
; CHECK-NOREGS-VP: LV(REG): Cost of 56 from 7 spills of Generic::VectorRC
; CHECK-NOREGS-VP-NEXT: Cost for VF vscale x 16: 81 (Estimated cost per lane: 5.1)
; CHECK-NOREGS-VP: LV: Selecting VF: vscale x 4.
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr i32, ptr %a, i64 %iv
  %load.a = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr i8, ptr %b, i64 %iv
  %load.b = load i8, ptr %gep.b, align 1
  %ext.b = zext i8 %load.b to i32
  %add = add i32 %load.a, %ext.b
  store i32 %add, ptr %gep.a, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1024
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { vscale_range(1,16) "target-features"="+sve" }
