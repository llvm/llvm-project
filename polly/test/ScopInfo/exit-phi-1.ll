; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -disable-output < %s
;
; Verify we only create one SAI object for up.3.ph as it is outside the SCoP.
;
; CHECK: Region: %for.body
;
; CHECK:         Arrays {
; CHECK-NEXT:        double MemRef_up_3_ph; // Element size 8
; CHECK-NEXT:        i32* MemRef_A[*]; // Element size 8
; CHECK-NEXT:    }
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: uwtable
define void @_ZN6soplex14SPxAggregateSM9eliminateERKNS_7SVectorEd(i32** nocapture readonly %A) {
entry:
  br label %for.cond.outer304

for.cond.outer304:                                ; preds = %if.else113, %if.then111, %entry
  %up.3.ph = phi double [ 0.000000e+00, %entry ], [ undef, %if.else113 ], [ undef, %if.then111 ]
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond.outer304
  %0 = load i32*, i32** %A, align 8
  %add = fadd double %up.3.ph, undef
  %val.i.i.i235 = getelementptr inbounds i32, i32* %0, i64 0
  br i1 false, label %if.else113, label %if.then111

if.then111:                                       ; preds = %for.body
  br label %for.cond.outer304

if.else113:                                       ; preds = %for.body
  br label %for.cond.outer304

for.end:                                          ; preds = %for.cond.outer304
  ret void
}
