; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s -check-prefix=SCOP
; RUN: opt %loadPolly -S -polly-codegen-ppcg < %s | FileCheck %s -check-prefix=HOST-IR

; REQUIRES: pollyacc

; SCOP:      Function: checkPrivatization
; SCOP-NEXT: Region: %for.body---%for.end
; SCOP-NEXT: Max Loop Depth:  1


; Check that kernel launch is generated in host IR.
; the declare would not be generated unless a call to a kernel exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)

;
;
;    void checkPrivatization(int A[], int B[], int C[], int control) {
;      int x;
;    #pragma scop
;      for (int i = 0; i < 1000; i++) {
;        x = 0;
;        if (control)
;          x += C[i];
;
;        B[i] = x * A[i];
;      }
;    #pragma endscop
;    }
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @checkPrivatization(i32* %A, i32* %B, i32* %C, i32 %control) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %entry.split, %if.end
  %indvars.iv = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %if.end ]
  %tobool = icmp eq i32 %control, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %tmp4 = load i32, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  %x.0 = phi i32 [ %tmp4, %if.then ], [ 0, %for.body ]
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp9 = load i32, i32* %arrayidx2, align 4
  %mul = mul nsw i32 %tmp9, %x.0
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %if.end
  ret void
}
