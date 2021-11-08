; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-target-instruction-cost=1 -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL: Checking a loop in "interleaved_access"
; CHECK:         The Smallest and Widest types: 64 / 64 bits
;
define void @interleaved_access(i8** %A, i64 %N) {
for.ph:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next.3, %for.body ], [ 0, %for.ph ]
  %tmp0 = getelementptr inbounds i8*, i8** %A, i64 %i
  store i8* null, i8** %tmp0, align 8
  %i.next.0 = add nuw nsw i64 %i, 1
  %tmp1 = getelementptr inbounds i8*, i8** %A, i64 %i.next.0
  store i8* null, i8** %tmp1, align 8
  %i.next.1 = add nsw i64 %i, 2
  %tmp2 = getelementptr inbounds i8*, i8** %A, i64 %i.next.1
  store i8* null, i8** %tmp2, align 8
  %i.next.2 = add nsw i64 %i, 3
  %tmp3 = getelementptr inbounds i8*, i8** %A, i64 %i.next.2
  store i8* null, i8** %tmp3, align 8
  %i.next.3 = add nsw i64 %i, 4
  %cond = icmp slt i64 %i.next.3, %N
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; For in-loop reductions with no loads or stores in the loop the widest type is
; determined by looking through the recurrences, which allows a sensible VF to be
; chosen. The following 3 cases check different combinations of widths.

; CHECK-LABEL: Checking a loop in "no_loads_stores_32"
; CHECK: The Smallest and Widest types: 4294967295 / 32 bits
; CHECK: Selecting VF: 4

define double @no_loads_stores_32(i32 %n) {
entry:
  br label %for.body

for.body:
  %s.09 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %conv = sitofp i32 %i.08 to float
  %conv1 = fpext float %conv to double
  %add = fadd double %s.09, %conv1
  %inc = add nuw i32 %i.08, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  %.lcssa = phi double [ %add, %for.body ]
  ret double %.lcssa
}

; CHECK-LABEL: Checking a loop in "no_loads_stores_16"
; CHECK: The Smallest and Widest types: 4294967295 / 16 bits
; CHECK: Selecting VF: 8

define double @no_loads_stores_16() {
entry:
  br label %for.body

for.body:
  %s.09 = phi double [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %i.08 = phi i16 [ 0, %entry ], [ %inc, %for.body ]
  %conv = sitofp i16 %i.08 to double
  %add = fadd double %s.09, %conv
  %inc = add nuw nsw i16 %i.08, 1
  %exitcond.not = icmp eq i16 %inc, 12345
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  %.lcssa = phi double [ %add, %for.body ]
  ret double %.lcssa
}

; CHECK-LABEL: Checking a loop in "no_loads_stores_8"
; CHECK: The Smallest and Widest types: 4294967295 / 8 bits
; CHECK: Selecting VF: 16

define float @no_loads_stores_8() {
entry:
  br label %for.body

for.body:
  %s.09 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %i.08 = phi i8 [ 0, %entry ], [ %inc, %for.body ]
  %conv = sitofp i8 %i.08 to float
  %add = fadd float %s.09, %conv
  %inc = add nuw nsw i8 %i.08, 1
  %exitcond.not = icmp eq i8 %inc, 12345
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  %.lcssa = phi float [ %add, %for.body ]
  ret float %.lcssa
}
