; RUN: opt %loadNPMPolly -disable-output -polly-invariant-load-hoisting \
; RUN: -polly-allow-dereference-of-all-function-parameters \
; RUN: '-passes=print<polly-function-scops>' < %s 2>&1 | FileCheck %s --check-prefix=SCOP

; RUN: opt %loadNPMPolly -S -polly-invariant-load-hoisting \
; RUN: -passes=polly-codegen < %s 2>&1 | FileCheck %s --check-prefix=CODE-RTC


; RUN: opt %loadNPMPolly -S -polly-invariant-load-hoisting \
; RUN: -polly-allow-dereference-of-all-function-parameters \
; RUN: -passes=polly-codegen < %s 2>&1 | FileCheck %s --check-prefix=CODE

; SCOP:      Function: hoge
; SCOP-NEXT: Region: %bb15---%bb37
; SCOP-NEXT: Max Loop Depth:  2
; SCOP-NEXT: Invariant Accesses: {
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp, tmp17, tmp28] -> { Stmt_bb29[i0] -> MemRef_arg1[0] };
; SCOP-NEXT:         Execution Context: [tmp, tmp17, tmp28] -> {  :  }
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             [tmp, tmp17, tmp28] -> { Stmt_bb27[] -> MemRef_arg[0] };
; SCOP-NEXT:         Execution Context: [tmp, tmp17, tmp28] -> {  :  }
; SCOP-NEXT: }

; Check that without the option `-polly-allow-dereference-of-all-function-parameters`
; we do generate the runtime check.
; CODE-RTC: polly.preload.cond:                               ; preds = %polly.preload.begin
; CODE-RTC-NEXT: br i1 %{{[a-zA-Z0-9\.]*}}, label %polly.preload.exec, label %polly.preload.merge

; Check that we don't generate a runtime check because we treat all
; parameters as dereferenceable.
; CODE-NOT: polly.preload.cond:                               ; preds = %polly.preload.begin
; CODE-NOT: br i1 %{{r1:[a-zA-Z0-9]*}}, label %polly.preload.exec, label %polly.preload.merge

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@global = external global i32

; Function Attrs: nounwind uwtable
define void @hoge(ptr noalias %arg, ptr noalias %arg1, ptr noalias %arg2, ptr %A) #0 {
bb:
  %tmp = load i32, ptr @global, align 4
  %tmp3 = icmp sgt i32 %tmp, 1
  br label %bb14

bb14:                                             ; preds = %bb
  br label %bb15

bb15:                                             ; preds = %bb14
  br i1 %tmp3, label %bb16, label %bb27

bb16:                                             ; preds = %bb15
  %tmp17 = load i32, ptr %arg1, align 4
  br label %bb18

bb18:                                             ; preds = %bb18, %bb16
  %tmp19 = phi i32 [ %tmp25, %bb18 ], [ 1, %bb16 ]
  %tmp20 = sext i32 %tmp19 to i64
  %tmp21 = add nsw i64 %tmp20, -1
  %tmp22 = getelementptr [0 x double], ptr %arg2, i64 0, i64 %tmp21
  store i64 undef, ptr %tmp22, align 8
  %tmp24 = icmp eq i32 %tmp19, %tmp17
  %tmp25 = add i32 %tmp19, 1
  br i1 %tmp24, label %bb26, label %bb18

bb26:                                             ; preds = %bb18
  br label %bb27

bb27:                                             ; preds = %bb26, %bb15
  %tmp28 = load i32, ptr %arg, align 4
  store float 42.0, ptr %A
  br label %bb29

bb29:                                             ; preds = %bb35, %bb27
  %tmp30 = load i32, ptr %arg1, align 4
  store float 42.0, ptr %A
  br label %bb31

bb31:                                             ; preds = %bb31, %bb29
  %tmp32 = phi i32 [ 1, %bb31 ], [ 1, %bb29 ]
  store float 42.0, ptr %A
  %tmp33 = icmp eq i32 %tmp32, %tmp30
  br i1 %tmp33, label %bb34, label %bb31

bb34:                                             ; preds = %bb31
  br label %bb35

bb35:                                             ; preds = %bb34
  %tmp36 = icmp eq i32 0, %tmp28
  br i1 %tmp36, label %bb37, label %bb29

bb37:                                             ; preds = %bb35
  ret void
}

attributes #0 = { nounwind uwtable }
