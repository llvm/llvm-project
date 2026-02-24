; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize -force-vector-width=4 -disable-output 2>&1 < %s | FileCheck %s
; REQUIRES: asserts

; CHECK-LABEL: LV: Checking a loop in 'test_assumed_bounds_type_mismatch'
; CHECK:       LV: Not vectorizing: Cannot vectorize uncountable loop.

define void @test_assumed_bounds_type_mismatch(ptr noalias %array, ptr readonly %pred, i32 %n) nosync nofree {
entry:
  %n_bytes = mul nuw nsw i32 %n, 2
  call void @llvm.assume(i1 true) [ "dereferenceable"(ptr %pred, i32 %n_bytes) ]
  %tc = sext i32 %n to i64
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %st.addr = getelementptr inbounds i16, ptr %array, i64 %iv
  %data = load i16, ptr %st.addr, align 2
  %inc = add nsw i16 %data, 1
  store i16 %inc, ptr %st.addr, align 2
  %ee.addr = getelementptr inbounds i16, ptr %pred, i64 %iv
  %ee.val = load i16, ptr %ee.addr, align 2
  %ee.cond = icmp sgt i16 %ee.val, 500
  br i1 %ee.cond, label %exit, label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %counted.cond = icmp eq i64 %iv.next, %tc
  br i1 %counted.cond, label %exit, label %for.body

exit:
  ret void
}

declare void @llvm.assume(i1)