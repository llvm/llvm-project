; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-scops -analyze -polly-import-jscop -polly-import-jscop-postfix=transformed < %s | FileCheck %s
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -S < %s | FileCheck %s --check-prefix=CODEGEN
;
; #define Ni 1056
; #define Nj 1056
; #define Nk 1024
;
; void create_arrays_heap(double beta, double A[Ni][Nk], double B[Ni][Nj]) {
;   int i,j,k;
;
;   for (i = 0; i < Ni; i++) {
;     for (j = 0; j < Nj; j++) {
;       for (k = 0; k < Nk; ++k) {
; 	B[i][j] = beta * A[i][k];
;       }
;     }
;   }
; }
;
; Check if the info from the JSON file has been analysed without errors.
; CHECK: Arrays {
; CHECK: double MemRef_A[*][1024]; // Element size 8
; CHECK: double MemRef_beta; // Element size 8
; CHECK: double MemRef_B[*][1056]; // Element size 8
; CHECK: double D[270336]; // Element size 8
; CHECK: double E[270336][200000]; // Element size 8
; CHECK: i64 F[270336]; // Element size 8
;
; Check if there are the 3 expected malloc calls with the right parameters at polly.start.
; 	%D : size(D) = product_all_dimensions*sizeof(type) = 270336*8 = 2162688 cast to double*
;	%E : size(E) = 270336*200000*8 = 432537600000 cast to double*
; 	%F : size(F) = 270336*8 = 2162688 cast to i64*
; CODEGEN: polly.start:
; CODEGEN: %malloccall = tail call i8* @malloc(i64 2162688)
; CODEGEN: %D = bitcast i8* %malloccall to double*
; CODEGEN: %malloccall1 = tail call i8* @malloc(i64 432537600000)
; CODEGEN: %E = bitcast i8* %malloccall1 to double*
; CODEGEN: %malloccall2 = tail call i8* @malloc(i64 2162688)
; CODEGEN: %F = bitcast i8* %malloccall2 to i64*
;
; Check if there are the 3 expected malloc calls with the right parameters at polly.exiting.
; 	Cast to i8* before freeing because malloc give us a i8 and free is waiting for a i8*
; CODEGEN: polly.exiting:
; CODEGEN: %12 = bitcast double* %D to i8*
; CODEGEN: tail call void @free(i8* %12)
; CODEGEN: %13 = bitcast double* %E to i8*
; CODEGEN: tail call void @free(i8* %13)
; CODEGEN: %14 = bitcast i64* %F to i8*
; CODEGEN: tail call void @free(i8* %14)
;
; Check if the new access for array E is present.
; CODEGEN: %polly.access.mul.E = mul nsw i64 %polly.indvar, 200000
; CODEGEN: %polly.access.add.E = add nsw i64 %polly.access.mul.E, %
; CODEGEN: %polly.access.E = getelementptr double, double* %E, i64 %polly.access.add.E
;
; ModuleID = 'create_arrays_heap.ll'
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @create_arrays_heap(double %beta, [1024 x double]* nocapture readonly %A, [1056 x double]* nocapture %B) local_unnamed_addr {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc16, %entry
  %indvars.iv35 = phi i64 [ 0, %entry ], [ %indvars.iv.next36, %for.inc16 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc13, %for.cond1.preheader
  %indvars.iv32 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next33, %for.inc13 ]
  %arrayidx12 = getelementptr inbounds [1056 x double], [1056 x double]* %B, i64 %indvars.iv35, i64 %indvars.iv32
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %indvars.iv = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next.3, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv35, i64 %indvars.iv
  %0 = load double, double* %arrayidx8, align 8
  %mul = fmul double %0, %beta
  store double %mul, double* %arrayidx12, align 8
  %indvars.iv.next = or i64 %indvars.iv, 1
  %arrayidx8.1 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv35, i64 %indvars.iv.next
  %1 = load double, double* %arrayidx8.1, align 8
  %mul.1 = fmul double %1, %beta
  store double %mul.1, double* %arrayidx12, align 8
  %indvars.iv.next.1 = or i64 %indvars.iv, 2
  %arrayidx8.2 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv35, i64 %indvars.iv.next.1
  %2 = load double, double* %arrayidx8.2, align 8
  %mul.2 = fmul double %2, %beta
  store double %mul.2, double* %arrayidx12, align 8
  %indvars.iv.next.2 = or i64 %indvars.iv, 3
  %arrayidx8.3 = getelementptr inbounds [1024 x double], [1024 x double]* %A, i64 %indvars.iv35, i64 %indvars.iv.next.2
  %3 = load double, double* %arrayidx8.3, align 8
  %mul.3 = fmul double %3, %beta
  store double %mul.3, double* %arrayidx12, align 8
  %indvars.iv.next.3 = add nsw i64 %indvars.iv, 4
  %exitcond.3 = icmp eq i64 %indvars.iv.next.3, 1024
  br i1 %exitcond.3, label %for.inc13, label %for.body6

for.inc13:                                        ; preds = %for.body6
  %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
  %exitcond34 = icmp eq i64 %indvars.iv.next33, 1056
  br i1 %exitcond34, label %for.inc16, label %for.cond4.preheader

for.inc16:                                        ; preds = %for.inc13
  %indvars.iv.next36 = add nuw nsw i64 %indvars.iv35, 1
  %exitcond37 = icmp eq i64 %indvars.iv.next36, 1056
  br i1 %exitcond37, label %for.end18, label %for.cond1.preheader

for.end18:                                        ; preds = %for.inc16
  ret void
}

