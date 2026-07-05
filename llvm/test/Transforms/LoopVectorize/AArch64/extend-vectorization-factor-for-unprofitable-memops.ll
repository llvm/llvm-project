; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-darwin -S %s | FileCheck %s

; Test cases for extending the vectorization factor, if small memory operations
; are not profitable.

; Test with a loop that contains memory accesses of i8 and i32 types. The
; maximum VF for NEON is calculated by 128/size of smallest type in loop.
; And while we don't have an instruction to  load 4 x i8, vectorization
; might still be profitable.
define void @test_load_i8_store_i32(ptr noalias %src, ptr noalias %dst, i32 %off, i64 %N) {
; CHECK-LABEL: @test_load_i8_store_i32(
; CHECK:       <16 x i8>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep.src = getelementptr inbounds i8, ptr %src, i64 %iv
  %lv = load i8, ptr %gep.src, align 1
  %lv.ext = zext i8 %lv to i32
  %add = add i32 %lv.ext, %off
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 %add, ptr %gep.dst
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; Same as test_load_i8_store_i32, but with types flipped for load and store.
define void @test_load_i32_store_i8(ptr noalias %src, ptr noalias %dst, i32 %off, i64 %N) {
; CHECK-LABEL: @test_load_i32_store_i8(
; CHECK:     <16 x i8>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %lv = load i32, ptr %gep.src, align 1
  %add = add i32 %lv, %off
  %add.trunc = trunc i32 %add to i8
  %gep.dst = getelementptr inbounds i8, ptr %dst, i64 %iv
  store i8 %add.trunc, ptr %gep.dst
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; All memory operations use i32, all memory operations are profitable with VF 4.
define void @test_load_i32_store_i32(ptr noalias %src, ptr noalias %dst, i8 %off, i64 %N) {
; CHECK-LABEL: @test_load_i32_store_i32(
; CHECK: vector.body:
; CHECK:   <4 x i32>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %lv = load i32, ptr %gep.src, align 1
  %lv.trunc = trunc i32 %lv to i8
  %add = add i8 %lv.trunc, %off
  %add.ext = zext i8 %add to i32
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 %add.ext, ptr %gep.dst
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}

; Test with loop body that requires a large number of vector registers if the
; vectorization factor is large. Make sure the register estimates limit the
; vectorization factor.
define void @test_load_i8_store_i64_large(ptr noalias %src, ptr noalias %dst, ptr noalias %dst.2, ptr noalias %dst.3, ptr noalias %dst.4, ptr noalias %dst.5, i64%off, i64 %off.2, i64 %N) {
; CHECK-LABEL: @test_load_i8_store_i64_large
; CHECK: <8 x i64>
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %gep.src = getelementptr inbounds i8, ptr %src, i64 %iv
  %gep.dst.3 = getelementptr inbounds i64, ptr %dst.3, i64 %iv
  %lv.dst.3 = load i64, ptr %gep.dst.3, align 1
  %gep.dst.5 = getelementptr inbounds i64, ptr %dst.5, i64 %iv
  %lv.dst.5 = load i64, ptr %gep.dst.3, align 1

  %lv = load i8, ptr %gep.src, align 1
  %lv.ext = zext i8 %lv to i64
  %add = add i64 %lv.ext, %off
  %add.2 = add i64 %add, %off.2
  %gep.dst = getelementptr inbounds i64, ptr %dst, i64 %iv
  %gep.dst.2 = getelementptr inbounds i64, ptr %dst.2, i64 %iv

  %add.3 = add i64 %add.2, %lv.dst.3
  %add.4 = add i64 %add.3, %add
  %gep.dst.4 = getelementptr inbounds i64, ptr %dst.4, i64 %iv
  %add.5 = add i64 %add.2, %lv.dst.5
  store i64 %add.2, ptr %gep.dst.2
  store i64 %add, ptr %gep.dst
  store i64 %add.3, ptr %gep.dst.3
  store i64 %add.4, ptr %gep.dst.4
  store i64 %add.5, ptr %gep.dst.5

  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}
