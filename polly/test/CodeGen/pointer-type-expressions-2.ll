; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s -check-prefix=CODEGEN
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i8* %start, i8* %end) {
entry:
  %A = alloca i32
  br label %body

body:
  %ptr = phi i8* [ %start, %entry ], [ %ptr2, %body ]
  %ptr2 = getelementptr inbounds i8, i8* %ptr, i64 1
  %cmp = icmp eq i8* %ptr2, %end
  store i32 42, i32* %A
  br i1 %cmp, label %exit, label %body

exit:
  ret void
}

; CHECK: for (int c0 = 0; c0 < -start + end; c0 += 1)
; CHECK:   Stmt_body(c0);

; CODEGEN-LABEL: polly.start:
; CODEGEN-NEXT:   %[[r0:[._a-zA-Z0-9]*]] = ptrtoint i8* %start to i64
; CODEGEN-NEXT:   %[[r1:[._a-zA-Z0-9]*]] = sub nsw i64 0, %[[r0]]
; CODEGEN-NEXT:   %[[r2:[._a-zA-Z0-9]*]] = ptrtoint i8* %end to i64
; CODEGEN-NEXT:   %[[r4:[._a-zA-Z0-9]*]] = add nsw i64 %[[r1]], %[[r2]]

