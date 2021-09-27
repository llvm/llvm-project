; RUN: opt %loadPolly -polly-codegen -polly-invariant-load-hoisting=true -S < %s | FileCheck %s

; CHECK: polly.start

;    void f(int *A) {
;      if (*A > 42)
;        *A = *A + 1;
;      else
;        *A = *A - 1;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
entry:
  br label %entry.split

entry.split:
  %tmp = load i32, i32* %A, align 4
  %cmp = icmp sgt i32 %tmp, 42
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %tmp1 = load i32, i32* %A, align 4
  %add = add nsw i32 %tmp1, 1
  br label %if.end

if.else:                                          ; preds = %entry
  %tmp2 = load i32, i32* %A, align 4
  %sub = add nsw i32 %tmp2, -1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %storemerge = phi i32 [ %sub, %if.else ], [ %add, %if.then ]
  store i32 %storemerge, i32* %A, align 4
  ret void
}
