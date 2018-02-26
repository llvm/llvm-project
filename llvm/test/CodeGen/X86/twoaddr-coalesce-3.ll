; RUN: llc < %s -mtriple=x86_64-- -relocation-model=pic | FileCheck %s
; This test is to ensure the TwoAddrInstruction pass chooses the proper operands to
; merge and generates fewer mov insns.

@M = common global i32 0, align 4
@total = common global i32 0, align 4
@g = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @foo() {
entry:
  %0 = load i32, i32* @M, align 4
  %cmp3 = icmp sgt i32 %0, 0
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %total.promoted = load i32, i32* @total, align 4
  br label %for.body

; Check that only one mov will be generated in the kernel loop.
; CHECK-LABEL: foo:
; CHECK: [[LOOP1:^[a-zA-Z0-9_.]+]]: {{#.*}} %for.body{{$}}
; CHECK-NOT: mov
; CHECK: movl {{.*}}, [[REG1:%[a-z0-9]+]]
; CHECK-NOT: mov
; CHECK: shrl $31, [[REG1]]
; CHECK-NOT: mov
; CHECK: jl [[LOOP1]]
for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %add5 = phi i32 [ %total.promoted, %for.body.lr.ph ], [ %add, %for.body ]
  %i.04 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %div = sdiv i32 %i.04, 2
  %add = add nsw i32 %div, %add5
  %inc = add nuw nsw i32 %i.04, 1
  %cmp = icmp slt i32 %inc, %0
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  store i32 %add, i32* @total, align 4
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

; Function Attrs: nounwind uwtable
define void @goo() {
entry:
  %0 = load i32, i32* @M, align 4
  %cmp3 = icmp sgt i32 %0, 0
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %total.promoted = load i32, i32* @total, align 4
  br label %for.body

; Check that only two mov will be generated in the kernel loop.
; CHECK-LABEL: goo:
; CHECK: [[LOOP2:^[a-zA-Z0-9_.]+]]: {{#.*}} %for.body{{$}}
; CHECK-NOT: mov
; CHECK: movl {{.*}}, [[REG2:%[a-z0-9]+]]
; CHECK-NOT: mov
; CHECK: shrl $31, [[REG2]]
; CHECK-NOT: mov
; CHECK: movl {{.*}}
; CHECK: jl [[LOOP2]]
for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %add5 = phi i32 [ %total.promoted, %for.body.lr.ph ], [ %add, %for.body ]
  %i.04 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %div = sdiv i32 %i.04, 2
  %add = add nsw i32 %div, %add5
  store volatile i32 %add, i32* @g, align 4
  %inc = add nuw nsw i32 %i.04, 1
  %cmp = icmp slt i32 %inc, %0
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  store i32 %add, i32* @total, align 4
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

