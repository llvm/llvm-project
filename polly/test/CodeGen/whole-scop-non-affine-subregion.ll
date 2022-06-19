; RUN: opt %loadPolly \
; RUN: -polly-codegen -S < %s | FileCheck %s

; CHECK: polly.start
;    int /* pure */ g()
;    void f(int *A) {
;      if (g())
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
  %call = call i32 @g()
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %tmp1 = load i32, i32* %A, align 4
  %add = add nsw i32 %tmp1, 1
  store i32 %add, i32* %A, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %tmp2 = load i32, i32* %A, align 4
  %sub = add nsw i32 %tmp2, -1
  store i32 %sub, i32* %A, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare i32 @g() #0

attributes #0 = { nounwind readnone }
