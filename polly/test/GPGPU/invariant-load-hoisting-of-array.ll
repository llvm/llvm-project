; RUN: opt %loadPolly -analyze -polly-scops \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=SCOP

; RUN: opt %loadPolly -S -polly-codegen-ppcg \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=HOST-IR

; REQUIRES: pollyacc

; Entry: Contains (%loaded.ptr.preload.s2a = alloca double*) which is
;   |    invariant load hoisted `%loaded.ptr`
;   v
; Run-time check --(failure branch)--> { old code - contains `%loaded.ptr` }
;   |
;  (success branch)
;   |
;   v
; New Code: Should refer to `%loaded.ptr.preload.s2a`, which is
;           the invariant load hoisted value, NOT `%loaded.ptr`.

; In Polly, we preserve the old code and create a separate branch that executes
; the GPU code if a run-time check succeeds.

; We need to make sure that in the new branch, we pick up invariant load hoisted
; values. The old values will belong to the old code branch.

; In this case, we use to try to load the 'original' %loaded.ptr in the
; 'New Code' branch,which is wrong. Check that this does not happen.

; Check that we have a Scop with an invariant load of the array.
; SCOP:       Function: f
; SCOP-NEXT:  Region: %arrload---%for.exit
; SCOP-NEXT:  Max Loop Depth:  1
; SCOP-NEXT:  Invariant Accesses: {
; SCOP-NEXT:          ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:              { Stmt_arrload[] -> MemRef_arr_of_ptrs[0] };



; Check that we have the preloaded array.
; HOST-IR: entry:
; HOST-IR-NEXT:  %loaded.ptr.preload.s2a = alloca double*

; Chek that we store the correct value in the preload.
; polly.preload.begin:                              ; preds = %polly.split_new_and_old
; HOST-IR: %polly.access.arr.of.ptrs = getelementptr double*, double** %arr.of.ptrs, i64 0
; HOST-IR-NEXT: %polly.access.arr.of.ptrs.load = load double*, double** %polly.access.arr.of.ptrs
; HOST-IR-NEXT: store double* %polly.access.arr.of.ptrs.load, double** %loaded.ptr.preload.s2a

; Check that we get back data from the kernel.
; HOST-IR: polly.acc.initialize:                             ; preds = %polly.start
; HOST-IR: [[FIRSTINDEX:%.+]] = getelementptr double, double* %polly.access.arr.of.ptrs.load, i64 1
; HOST-IR: [[BITCASTED:%.+]] = bitcast double* [[FIRSTINDEX]] to i8*
; HOST-IR: call void @polly_copyFromDeviceToHost(i8* %p_dev_array_MemRef_loaded_ptr, i8* [[BITCASTED]], i64 800)

; Check that the kernel launch is generated in the host IR.
; This declaration would not have been generated unless a kernel launch exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)


; C pseudocode equivalent
; void f(double **arr_of_ptrs) {
;     double *loaded_ptr = arr_of_ptrs[0];
;     if (false) { return; }
;     else {
;         for(int i = 1; i < 100; i++) {
;             loaded_ptr[i] = 42.0;
;         }
;     }
; }


target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"


; Function Attrs: nounwind uwtable
define void @f(double **%arr.of.ptrs) #0 {
entry:
  br label %arrload

arrload:                                             ; preds = %"7"
  %loaded.ptr = load double*, double** %arr.of.ptrs, align 8
  br i1 false, label %"for.exit", label %"for.preheader"

"for.preheader":                                       ; preds = %"51"
  br label %"for.body"

"for.body":                                             ; preds = %"53", %"53.lr.ph"
  %indvar = phi i64 [ 1, %"for.preheader" ], [ %indvar.next, %"for.body" ]
  %slot = getelementptr double, double* %loaded.ptr, i64 %indvar
  store double 42.0, double* %slot, align 8

  %indvar.next = add nuw nsw i64 %indvar, 1

  %check = icmp sgt i64 %indvar.next, 100
  br i1 %check, label %"for.exit", label %"for.body"

"for.exit":                                             ; preds = %"52.54_crit_edge", %"51"
    ret void
}

attributes #0 = { nounwind uwtable }
