; RUN: opt < %s -globals-aa -gvn -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@a = internal global i32* null, align 8
@b = global i32** @a, align 8
@c = global i32** @a, align 8
@d = common global i32 0, align 4

; Make sure we globals-aa doesn't get confused and allow hoisting
; the load from @a out of the loop.

; CHECK-LABEL: define i32 @main()
; CHECK: for.body:
; CHECK-NEXT:   %2 = load i32**, i32*** @b, align 8
; CHECK-NEXT:   store i32* @d, i32** %2, align 8
; CHECK-NEXT:   %3 = load i32*, i32** @a, align 8
; CHECK-NEXT:   %cmp1 = icmp ne i32* %3, @d
; CHECK-NEXT:   br i1 %cmp1, label %if.then, label %if.end

define i32 @main() {
entry:
  %0 = load i32, i32* @d, align 4
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %1 = phi i32 [ %inc, %if.end ], [ %0, %entry ]
  %cmp = icmp slt i32 %1, 1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32**, i32*** @b, align 8
  store i32* @d, i32** %2, align 8
  %3 = load i32*, i32** @a, align 8
  %cmp1 = icmp ne i32* %3, @d
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  br label %return

if.end:                                           ; preds = %for.body
  %4 = load i32, i32* @d, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, i32* @d, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %return

return:                                           ; preds = %for.end, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %for.end ]
  ret i32 %retval.0
}

