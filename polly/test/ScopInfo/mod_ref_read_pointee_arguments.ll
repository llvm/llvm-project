; RUN: opt %loadPolly -basicaa -polly-stmt-granularity=bb -polly-scops -analyze -polly-allow-modref-calls \
; RUN: < %s | FileCheck %s
; RUN: opt %loadPolly -basicaa -polly-codegen -disable-output \
; RUN: -polly-allow-modref-calls < %s
;
; Verify that we model the read access of the gcread intrinsic
; correctly, thus that A is read by it but B is not.
;
; CHECK:      Stmt_for_body
; CHECK-NEXT:   Domain :=
; CHECK-NEXT:       { Stmt_for_body[i0] : 0 <= i0 <= 1023 };
; CHECK-NEXT:   Schedule :=
; CHECK-NEXT:       { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:   ReadAccess := [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_A[o0] };
; CHECK-NEXT:   MustWriteAccess :=  [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_dummyloc[0] };
; CHECK-NEXT:   ReadAccess := [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_B[i0] };
; CHECK-NEXT:   MustWriteAccess :=  [Reduction Type: NONE]
; CHECK-NEXT:       { Stmt_for_body[i0] -> MemRef_A[i0] };
;
;    void jd(int *restirct A, int *restrict B) {
;      char **dummyloc;
;      for (int i = 0; i < 1024; i++) {
;        char *dummy = @llvm.gcread(A, nullptr);
;        *dummyloc = dummy;
;        A[i] = B[i];
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* noalias %A, i32* noalias %B) gc "dummy" {
entry:
  %dummyloc = alloca i8*
  br label %entry.split

entry.split:					  ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %for.inc
  %i = phi i64 [ 0, %entry.split ], [ %i.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i
  %arrayidx1 = getelementptr inbounds i32, i32* %B, i64 %i
  %bc = bitcast i32* %arrayidx to i8*
  %dummy = call i8* @f(i8* %bc, i8** null)
  store i8* %dummy, i8** %dummyloc, align 4
  %tmp = load i32, i32* %arrayidx1
  store i32 %tmp, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

declare i8* @f(i8*, i8**) #0

attributes #0 = { argmemonly readonly nounwind }
