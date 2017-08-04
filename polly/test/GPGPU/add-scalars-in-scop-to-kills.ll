; RUN: opt %loadPolly -analyze -polly-scops < %s | FileCheck %s -check-prefix=SCOP
; RUN: opt %loadPolly -S -polly-codegen-ppcg < %s | FileCheck %s -check-prefix=HOST-IR

; REQUIRES: pollyacc

; Check that we detect a scop.
; SCOP:       Function: checkScalarKill
; SCOP-NEXT: Region: %XLoopInit---%for.end
; SCOP-NEXT: Max Loop Depth:  1

; Check that we have a scalar that is not a phi node in the scop.
; SCOP: i32 MemRef_x_0; // Element size 4

; Check that kernel launch is generated in host IR.
; the declare would not be generated unless a call to a kernel exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)

; Check that we add variables that are local to a scop into the kills that we
; pass to PPCG. This should enable PPCG to codegen this example.
; void checkScalarKill(int A[], int B[], int C[], const int control1, int control2) {
; int x;
; #pragma scop
;     for(int i = 0; i < 1000; i++) {
; XLoopInit:        x = 0;
; 
;         if (control1 > 2)
;             C1Add: x += 10;
;         if (control2 > 3)
;             C2Add: x += A[i];
; 
; BLoopAccumX:        B[i] += x;
;     }
; 
; #pragma endscop
; }
; ModuleID = 'test.ll'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @checkScalarKill(i32* %A, i32* %B, i32* %C, i32 %control1, i32 %control2) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %XLoopInit

XLoopInit:                                        ; preds = %entry.split, %BLoopAccumX
  %indvars.iv = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %BLoopAccumX ]
  %cmp1 = icmp sgt i32 %control1, 2
  %x.0 = select i1 %cmp1, i32 10, i32 0
  %cmp2 = icmp sgt i32 %control2, 3
  br i1 %cmp2, label %C2Add, label %BLoopAccumX

C2Add:                                            ; preds = %XLoopInit
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp6 = load i32, i32* %arrayidx, align 4
  %add4 = add nsw i32 %tmp6, %x.0
  br label %BLoopAccumX

BLoopAccumX:                                      ; preds = %XLoopInit, %C2Add
  %x.1 = phi i32 [ %add4, %C2Add ], [ %x.0, %XLoopInit ]
  %arrayidx7 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %tmp11 = load i32, i32* %arrayidx7, align 4
  %add8 = add nsw i32 %tmp11, %x.1
  store i32 %add8, i32* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %XLoopInit, label %for.end

for.end:                                          ; preds = %BLoopAccumX
  ret void
}
