; RUN: opt %loadNPMPolly -passes='sroa,instcombine,simplifycfg,reassociate,loop(loop-rotate),instcombine,indvars,polly-prepare,print<polly-function-scops>' \
; RUN:    -tailcallopt -disable-output < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=NOLICM

; RUN: opt %loadNPMPolly -passes='sroa,instcombine,simplifycfg,reassociate,loop(loop-rotate),instcombine,indvars,loop-mssa(licm),polly-prepare,print<polly-function-scops>' \
; RUN:    -tailcallopt -disable-output < %s 2>&1 \
; RUN:     | FileCheck %s --check-prefix=LICM

;    void foo(int n, float A[static const restrict n], float x) {
;      //      (0)
;      for (int i = 0; i < 5; i += 1) {
;        for (int j = 0; j < n; j += 1) {
;          x = 7; // (1)
;        }
;        A[0] = x; // (3)
;      }
;      // (4)
;    }

; LICM:   Statements
; NOLICM: Statements

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i32 %n, ptr noalias nonnull %A, float %x) {
entry:
  %n.addr = alloca i32, align 4
  %A.addr = alloca ptr, align 8
  %x.addr = alloca float, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store ptr %A, ptr %A.addr, align 8
  store float %x, ptr %x.addr, align 4
  %tmp = load i32, ptr %n.addr, align 4
  %tmp1 = zext i32 %tmp to i64
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc.4, %entry
  %tmp2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %tmp2, 5
  br i1 %cmp, label %for.body, label %for.end.6

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %j, align 4
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc, %for.body
  %tmp3 = load i32, ptr %j, align 4
  %tmp4 = load i32, ptr %n.addr, align 4
  %cmp2 = icmp slt i32 %tmp3, %tmp4
  br i1 %cmp2, label %for.body.3, label %for.end

for.body.3:                                       ; preds = %for.cond.1
  store float 7.000000e+00, ptr %x.addr, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body.3
  %tmp5 = load i32, ptr %j, align 4
  %add = add nsw i32 %tmp5, 1
  store i32 %add, ptr %j, align 4
  br label %for.cond.1

for.end:                                          ; preds = %for.cond.1
  %tmp6 = load float, ptr %x.addr, align 4
  %tmp7 = load ptr, ptr %A.addr, align 8
  store float %tmp6, ptr %tmp7, align 4
  br label %for.inc.4

for.inc.4:                                        ; preds = %for.end
  %tmp8 = load i32, ptr %i, align 4
  %add5 = add nsw i32 %tmp8, 1
  store i32 %add5, ptr %i, align 4
  br label %for.cond

for.end.6:                                        ; preds = %for.cond
  ret void
}

; CHECK: Statements {
; CHECK:     Stmt_for_end
; CHECK: }
