; REQUIRES: asserts
; RUN: opt -mtriple=s390x-unknown-linux -mcpu=z13 -passes=loop-vectorize \
; RUN:   -debug-only=loop-vectorize \
; RUN:   -disable-output -enable-interleaved-mem-accesses=false < %s 2>&1 | \
; RUN:   FileCheck %s
;
; Check that a scalarized load/store does not get a cost for insterts/
; extracts, since z13 supports element load/store.

define void @fun(ptr %data, i64 %n) {
; CHECK-LABEL: LV: Checking a loop in 'fun'
; CHECK: LV: Scalarizing:  %tmp1 = load i32, ptr %tmp0, align 4
; CHECK: LV: Scalarizing:  store i32 %tmp2, ptr %tmp0, align 4

; CHECK: Cost of 4 for VF 4: REPLICATE ir<%tmp1> = load ir<%tmp0>
; CHECK: Cost of 4 for VF 4: REPLICATE store ir<%tmp2>, ir<%tmp0>
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %tmp0 = getelementptr inbounds i32, ptr %data, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  %tmp2 = add i32 %tmp1, 1
  store i32 %tmp2, ptr %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 2
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

define void @predicated_store(ptr noalias %dst, ptr %src.float, ptr %src.i32.0, ptr %src.i32.1, i64 %n) #0 {
; CHECK-LABEL: LV: Checking a loop in 'predicated_store'
; CHECK: Cost of 0 for VF 2: REPLICATE ir<%load.0> = load ir<%gep.0>
; CHECK: Cost of 0 for VF 2: REPLICATE store ir<0>, ir<%dst>
; CHECK: Cost of 0 for VF 4: REPLICATE ir<%load.0> = load ir<%gep.0>
; CHECK: Cost of 0 for VF 4: REPLICATE store ir<0>, ir<%dst>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop.latch ], [ 0, %entry ]
  %gep.0 = getelementptr i32, ptr %src.i32.0, i64 %iv
  %load.0 = load i32, ptr %gep.0, align 4
  %gep.1 = getelementptr i32, ptr %src.i32.1, i64 %iv
  %ext = sext i32 %load.0 to i64
  %mul = mul i64 %n, %ext
  %gep.float = getelementptr float, ptr %src.float, i64 %mul
  %load.float = load float, ptr %gep.float, align 4
  %fcmp = fcmp ogt float %load.float, 0.000000e+00
  %load.1 = load i32, ptr %gep.1, align 4
  %icmp = icmp sgt i32 %load.1, 0
  %cond = and i1 %fcmp, %icmp
  br i1 %cond, label %if.then, label %loop.latch

if.then:
  store i32 0, ptr %dst, align 4
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

attributes #0 = { "target-cpu" = "z16" }
