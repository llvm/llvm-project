; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -passes=loop-vectorize -debug-only=loop-vectorize -S < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; CHECK: 'foo'
; CHECK: LV: Found an estimated cost of 1 for VF 1 For instruction:   %shift = ashr i32 %val, %k
; CHECK: Cost of 2 for VF 2: WIDEN ir<%shift> = ashr ir<%val>, ir<%k>
; CHECK: Cost of 2 for VF 4: WIDEN ir<%shift> = ashr ir<%val>, ir<%k>
define void @foo(ptr nocapture %p, i32 %k) local_unnamed_addr {
entry:
  br label %body

body:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %ptr = getelementptr inbounds i32, ptr %p, i64 %i
  %val = load i32, ptr %ptr, align 4
  %shift = ashr i32 %val, %k
  store i32 %shift, ptr %ptr, align 4
  %next = add nuw nsw i64 %i, 1
  %cmp = icmp eq i64 %next, 16
  br i1 %cmp, label %exit, label %body

exit:
  ret void
}

; CHECK: 'shift_and_masked_load_store'
; CHECK: Cost of 1 for VF 2: CLONE ir<%shifted> = lshr vp<{{.+}}>, ir<2>
; CHECK: Cost of 1 for VF 4: CLONE ir<%shifted> = lshr vp<{{.+}}>, ir<2>
; CHECK: Cost of 4 for VF 8: WIDEN ir<%shifted> = lshr ir<%iv>, ir<2>
define void @shift_and_masked_load_store(i64 %trip.count) #0 {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %shifted = lshr i64 %iv, 2
  %masked.idx = and i64 %shifted, 1
  %load.ptr = getelementptr i16, ptr poison, i64 %masked.idx
  %val = load i16, ptr %load.ptr, align 2
  %store.idx = shl nuw i64 %iv, 2
  %store.ptr = getelementptr i8, ptr poison, i64 %store.idx
  store i16 %val, ptr %store.ptr, align 2
  %iv.next = add i64 %iv, 1
  %cmp = icmp eq i64 %iv, %trip.count
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}

define i64 @sdiv_arg_outer_iv(ptr noalias %dst, ptr %src) {
; CHECK: 'sdiv_arg_outer_iv'
; CHECK: Cost of 0 for VF 2: CLONE ir<%div> = sdiv ir<%add.offset>, ir<8>
; CHECK: Cost of 0 for VF 4: CLONE ir<%div> = sdiv ir<%add.offset>, ir<8>
; CHECK: Cost of 0 for VF 8: CLONE ir<%div> = sdiv ir<%add.offset>, ir<8>
; CHECK: Cost of 0 for VF 16: REPLICATE ir<%div> = sdiv ir<%add.offset>, ir<8>
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i32 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  %offset = shl nsw i32 %outer.iv, 7
  br label %loop

loop:
  %iv = phi i64 [ 0, %outer.header ], [ %iv.next, %loop ]
  %iv.trunc = trunc i64 %iv to i32
  %add.offset = add i32 %offset, %iv.trunc
  %div = sdiv i32 %add.offset, 8
  %div.ext = sext i32 %div to i64
  %gep.src = getelementptr i8, ptr %src, i64 %div.ext
  %l = load i8, ptr %gep.src, align 1
  %gep.dst = getelementptr i8, ptr %dst, i64 %iv
  store i8 %l, ptr %gep.dst, align 1
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv, 64
  br i1 %ec, label %outer.latch, label %loop

outer.latch:
  %outer.iv.next = add nsw i32 %outer.iv, 1
  br label %outer.header
}

attributes #0 = { "target-features"="+avx2" "tune-cpu"="alderlake" }
