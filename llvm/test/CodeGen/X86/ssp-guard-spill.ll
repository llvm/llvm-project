; RUN: llc < %s | FileCheck %s
target triple = "x86_64-apple-macosx10.4.0"

; __stack_chk_guard must be loaded for twice, once for setting up the canary,
; another time for performing the check. It is because if we reuse the same
; stack guard value, it may get spilled to the stack, then the for loop may
; corrupt it.
;
; bool Bar(int*);
; bool Foo(int n) {
;   int a[10];
;   for (int i = 0; i < n; i++) {
;     a[i] = 0;
;   }
;   return Bar(a);
; }
;
; CHECK: movq ___stack_chk_guard
; CHECK: movq ___stack_chk_guard
define zeroext i1 @_Z3Fooi(i32 %n) sspstrong {
entry:
  %n.addr = alloca i32, align 4
  %a = alloca [10 x i32], align 16
  %i = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %tmp = load i32, i32* %i, align 4
  %tmp1 = load i32, i32* %n.addr, align 4
  %cmp = icmp slt i32 %tmp, %tmp1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp2 = load i32, i32* %i, align 4
  %idxprom = sext i32 %tmp2 to i64
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 %idxprom
  store i32 0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %tmp3 = load i32, i32* %i, align 4
  %inc = add nsw i32 %tmp3, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arraydecay = getelementptr inbounds [10 x i32], [10 x i32]* %a, i32 0, i32 0
  %call = call zeroext i1 @_Z3BarPi(i32* %arraydecay)
  ret i1 %call
}

declare zeroext i1 @_Z3BarPi(i32*)
