; RUN: opt %loadPolly -polly-invariant-load-hoisting -polly-detect-full-functions -polly-print-scops -disable-output < %s | FileCheck %s
;
; This testcase checks for compatibility of the -detect-full-functions
; flag in combination with the -invariant-load-hoisting option. More
; specifically, ScopHelper.cpp::isHoistableLoad only gets called if
; -invariant-load-hoisting is enabled. This function, however, had a bug
; which caused a crash if the region argument was top-level. This test
; is a minimal example that hits this specific code path.
;
; Also note that this file's IR is in no way optimized, i.e. it was
; generated with clang -O0 from the following C-code:
;
;    void test() {
;      int A[] = {1, 2, 3, 4, 5};
;      int len = (sizeof A) / sizeof(int);
;      for (int i = 0; i < len; ++i) {
;        A[i] = A[i] * 2;
;      }
;    }
;
; This is also the reason why polly does not detect any scops (the loop
; variable i is loaded from and stored to memory in each iteration):
;
; CHECK:      region: 'for.cond => for.end' in function 'test':
; CHECK-NEXT: Invalid Scop!
; CHECK-NEXT: region: 'entry => <Function Return>' in function 'test':
; CHECK-NEXT: Invalid Scop!
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@test.A = private unnamed_addr constant [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5], align 16

define void @test() {
entry:
  %A = alloca [5 x i32], align 16
  %len = alloca i32, align 4
  %i = alloca i32, align 4
  %0 = bitcast [5 x i32]* %A to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast ([5 x i32]* @test.A to i8*), i64 20, i32 16, i1 false)
  store i32 5, i32* %len, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4
  %2 = load i32, i32* %len, align 4
  %cmp = icmp slt i32 %1, %2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [5 x i32], [5 x i32]* %A, i64 0, i64 %idxprom
  %4 = load i32, i32* %arrayidx, align 4
  %mul = mul nsw i32 %4, 2
  %5 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %5 to i64
  %arrayidx2 = getelementptr inbounds [5 x i32], [5 x i32]* %A, i64 0, i64 %idxprom1
  store i32 %mul, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %6 = load i32, i32* %i, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)
