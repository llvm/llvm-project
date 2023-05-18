; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-codegen -S < %s | FileCheck %s --check-prefix=CODEGEN
;
; Check for correct code generation of exit PHIs, even if the same PHI value
; is used again inside the the SCoP.
; Note that if.else113 is removed from the SCoP because it is never executed.
;
; CHECK: Region: %for.body
;
; CHECK:         Arrays {
; CHECK-NEXT:        double MemRef_up_3_ph; // Element size 8
; CHECK-NEXT:        ptr MemRef_A[*]; // Element size 8
; CHECK-NEXT:        double MemRef_up_3_ph; // Element size 8
; CHECK-NEXT:    }
;
; CODEGEN:      polly.merge_new_and_old:
; CODEGEN-NEXT:   %up.3.ph.ph.merge = phi double [ %up.3.ph.ph.final_reload, %polly.exiting ], [ undef, %for.cond.outer304.region_exiting ]
;
; CODEGEN:      for.cond.outer304:
; CODEGEN-NEXT:   %indvar = phi i64 [ %indvar.next, %polly.merge_new_and_old ], [ 0, %entry ]
; CODEGEN-NEXT:   %up.3.ph = phi double [ 0.000000e+00, %entry ], [ %up.3.ph.ph.merge, %polly.merge_new_and_old ]
;
; CODEGEN:      polly.stmt.if.then111:
; CODEGEN-NEXT:   store double undef, ptr %up.3.ph.s2a
;
; CODEGEN:      polly.exiting:
; CODEGEN-NEXT:   %up.3.ph.ph.final_reload = load double, ptr %up.3.ph.s2a
;
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: uwtable
define void @_ZN6soplex14SPxAggregateSM9eliminateERKNS_7SVectorEd(ptr nocapture readonly %A) {
entry:
  br label %for.cond.outer304

for.cond.outer304:                                ; preds = %if.else113, %if.then111, %entry
  %up.3.ph = phi double [ 0.000000e+00, %entry ], [ undef, %if.else113 ], [ undef, %if.then111 ]
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond.outer304
  %0 = load ptr, ptr %A, align 8
  %add = fadd double %up.3.ph, undef
  br i1 false, label %if.else113, label %if.then111

if.then111:                                       ; preds = %for.body
  br label %for.cond.outer304

if.else113:                                       ; preds = %for.body
  br label %for.cond.outer304

for.end:                                          ; preds = %for.cond.outer304
  ret void
}
