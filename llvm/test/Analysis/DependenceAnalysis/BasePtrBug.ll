; RUN: opt < %s -analyze -basicaa -da  | FileCheck %s

; Test that the dependence analysis generates the correct results when using
; an aliased object that points to a different element in the same array.
; PR33567 - https://bugs.llvm.org/show_bug.cgi?id=33567

; void test1(int *A, int *B, int N) {
;   int *top = A;
;   int *bot = A + N/2;
;   for (int i = 0; i < N; i++)
;     B[i] = top[i] + bot[i];
; }

; CHECK-LABEL: test1
; CHECK: da analyze - input [*|<]!

define void @test1(i32* nocapture %A, i32* nocapture %B, i32 %N) #0 {
entry:
  %cmp9 = icmp sgt i32 %N, 0
  br i1 %cmp9, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %div = sdiv i32 %N, 2
  %bot.gep = getelementptr i32, i32* %A, i32 %div
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %gep.0 = getelementptr i32, i32* %A, i32 %i
  %gep.1 = getelementptr i32, i32* %bot.gep, i32 %i
  %gep.B = getelementptr i32, i32* %B, i32 %i
  %0 = load i32, i32* %gep.0, align 4
  %1 = load i32, i32* %gep.1, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, i32* %gep.B, align 4
  %inc = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}


; void test2(int *A, unsigned n) {
;   int *B = A + 1;
;   for (unsigned i = 0; i < n; ++i) {
;     A[i] = B[i];
;   }
; }

; CHECK-LABEL: test2
; CHECK: da analyze - consistent anti [1]!

define void @test2(i32*, i32) #3 {
  %3 = getelementptr inbounds i32, i32* %0, i64 1
  br label %4

; <label>:4:
  %.0 = phi i32 [ 0, %2 ], [ %14, %13 ]
  %5 = sub i32 %1, 1
  %6 = icmp ult i32 %.0, %5
  br i1 %6, label %7, label %15

; <label>:7:
  %8 = zext i32 %.0 to i64
  %9 = getelementptr inbounds i32, i32* %3, i64 %8
  %10 = load i32, i32* %9, align 4
  %11 = zext i32 %.0 to i64
  %12 = getelementptr inbounds i32, i32* %0, i64 %11
  store i32 %10, i32* %12, align 4
  br label %13

; <label>:13:
  %14 = add i32 %.0, 1
  br label %4

; <label>:15:
  ret void
}
