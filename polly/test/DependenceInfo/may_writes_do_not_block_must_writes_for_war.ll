; RUN: opt %loadNPMPolly '-passes=print<polly-dependences>' -disable-output < %s | FileCheck %s
;
; Verify that the presence of a may-write (S1) between a read (S0) and a
; must-write (S2) does not block the generation of RAW dependences. This makes
; sure that we capture as many RAW dependences as possible.
;
; For this example, we want both (S0(Read) -> S1 (May-Write)) as well as
; (S0(Read) -> S2(Must-Write)).
;
; CHECK: WAR dependences:
; CHECK-NEXT:     { Stmt_S0[i0] -> Stmt_if_end__TO__S2[i0] : 0 < i0 <= 2; Stmt_S0[i0] -> Stmt_S2[i0] : 0 < i0 <= 2 }
;
;
;    static const int N = 3000;
;
;    void f(int *sum, int *A, int *B, int *out) {
;      for (int i = 0; i <= 2; i++) {
;        if (i) {
; S0:          *out += *sum;
;        }
;
;        if (i * i) {
; S1:          *sum = *A;
;        }
; S2:        *sum = *B;
;      }
;    }
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %sum, ptr %A, ptr %B, ptr %out) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 3
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tobool = icmp eq i32 %i.0, 0
  br i1 %tobool, label %if.end, label %S0

S0:                                          ; preds = %for.body
  %tmp = load i32, ptr %sum, align 4
  %tmp1 = load i32, ptr %out, align 4
  %add = add nsw i32 %tmp1, %tmp
  store i32 %add, ptr %out, align 4
  br label %if.end

if.end:                                           ; preds = %for.body, %S0
  %mul = mul nsw i32 %i.0, %i.0
  %tobool1 = icmp eq i32 %mul, 0
  br i1 %tobool1, label %S2, label %S1

S1:                                         ; preds = %if.end
  %tmp2 = load i32, ptr %A, align 4
  store i32 %tmp2, ptr %sum, align 4
  br label %S2

S2:                                          ; preds = %if.end, %S1
  %tmp3 = load i32, ptr %B, align 4
  store i32 %tmp3, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %S2
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

