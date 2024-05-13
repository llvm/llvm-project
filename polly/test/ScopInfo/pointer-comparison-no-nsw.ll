; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(int *A, int *B) {
;      while (A != B) {
;        *A = *A + 1;
;        A++;
;      }
;    }
;
; CHECK:      Invalid Context:
; CHECK-NEXT:  [A, B] -> { : (4*floor((A - B)/4) < A - B) or ((-A + B) mod 4 = 0 and B >= 9223372036854775808 + A) or ((-A + B) mod 4 = 0 and B <= -4 + A) }
;
; CHECK:      Domain :=
; CHECK-NEXT:   [A, B] -> { Stmt_while_body[i0] : (-A + B) mod 4 = 0 and i0 >= 0 and 4i0 <= -4 - A + B }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, ptr %B) {
entry:
  br label %entry.split

entry.split:
  br i1 true, label %while.cond, label %while.end

while.cond:                                       ; preds = %while.body, %entry
  %A.addr.0 = phi ptr [ %A, %entry.split ], [ %incdec.ptr, %while.body ]
  %cmp = icmp eq ptr %A.addr.0, %B
  br i1 %cmp, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %tmp = load i32, ptr %A.addr.0, align 4
  %add = add i32 %tmp, 1
  store i32 %add, ptr %A.addr.0, align 4
  %incdec.ptr = getelementptr i32, ptr %A.addr.0, i64 1
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}
