; RUN: opt %loadNPMPolly '-passes=polly-custom<scops>' -plugin-arg=Polly,-polly-print-scops -disable-output < %s 2>&1 | FileCheck %s
;
; Verify that an llvm.assume inside an interior block of a non-affine
; subregion does not cause an assertion failure in addUserAssumptions.
; The assume's block has no InvalidDomainMap entry and must be skipped.
;
; https://github.com/llvm/llvm-project/issues/226718
;
;    void test(int *A, int *B, long n, int m) {
;      for (long i = 0; i < n; i++) {
;        A[i] = m;
;        if (B[i]) {
;          if (!(m == 0)) { __builtin_unreachable(); }
;          B[i] = 1;
;        }
;      }
;    }
;
; CHECK: Printing analysis
; CHECK: Region: %for.body---%exit
; CHECK: Statements {
; CHECK: }

define void @test(ptr %A, ptr %B, i64 %n, i32 %m) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %for.body.ph, label %exit

for.body.ph:
  %cmp2 = icmp eq i32 %m, 0
  br label %for.body

for.body:
  %i = phi i64 [ 0, %for.body.ph ], [ %inc, %for.inc ]
  %arrayidx.A = getelementptr inbounds i32, ptr %A, i64 %i
  store i32 %m, ptr %arrayidx.A, align 4
  %arrayidx.B = getelementptr inbounds i32, ptr %B, i64 %i
  %val = load i32, ptr %arrayidx.B, align 4
  %tobool = icmp eq i32 %val, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:
  call void @llvm.assume(i1 %cmp2)
  store i32 1, ptr %arrayidx.B, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %inc, %n
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

declare void @llvm.assume(i1)
