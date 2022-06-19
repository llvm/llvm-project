; RUN: opt %loadPolly -basic-aa -sroa -instcombine -simplifycfg -tailcallopt \
; RUN:    -simplifycfg -reassociate -loop-rotate -instcombine -indvars \
; RUN:    -polly-prepare -polly-print-scops -disable-output < %s \
; RUN:     | FileCheck %s --check-prefix=NOLICM

; RUN: opt %loadPolly -basic-aa -sroa -instcombine -simplifycfg -tailcallopt \
; RUN:    -simplifycfg -reassociate -loop-rotate -instcombine -indvars -licm \
; RUN:    -polly-prepare -polly-print-scops -disable-output < %s \
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

define void @foo(i32 %n, float* noalias nonnull %A, float %x) {
entry:
  %n.addr = alloca i32, align 4
  %A.addr = alloca float*, align 8
  %x.addr = alloca float, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  store float* %A, float** %A.addr, align 8
  store float %x, float* %x.addr, align 4
  %tmp = load i32, i32* %n.addr, align 4
  %tmp1 = zext i32 %tmp to i64
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc.4, %entry
  %tmp2 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %tmp2, 5
  br i1 %cmp, label %for.body, label %for.end.6

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc, %for.body
  %tmp3 = load i32, i32* %j, align 4
  %tmp4 = load i32, i32* %n.addr, align 4
  %cmp2 = icmp slt i32 %tmp3, %tmp4
  br i1 %cmp2, label %for.body.3, label %for.end

for.body.3:                                       ; preds = %for.cond.1
  store float 7.000000e+00, float* %x.addr, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body.3
  %tmp5 = load i32, i32* %j, align 4
  %add = add nsw i32 %tmp5, 1
  store i32 %add, i32* %j, align 4
  br label %for.cond.1

for.end:                                          ; preds = %for.cond.1
  %tmp6 = load float, float* %x.addr, align 4
  %tmp7 = load float*, float** %A.addr, align 8
  %arrayidx = getelementptr inbounds float, float* %tmp7, i64 0
  store float %tmp6, float* %arrayidx, align 4
  br label %for.inc.4

for.inc.4:                                        ; preds = %for.end
  %tmp8 = load i32, i32* %i, align 4
  %add5 = add nsw i32 %tmp8, 1
  store i32 %add5, i32* %i, align 4
  br label %for.cond

for.end.6:                                        ; preds = %for.cond
  ret void
}

; CHECK: Statements {
; CHECK:     Stmt_for_end
; CHECK: }
