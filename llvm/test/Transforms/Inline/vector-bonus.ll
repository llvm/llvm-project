; RUN: opt < %s -passes=inline -inline-threshold=35  -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -inline-threshold=35  -S | FileCheck %s

define i32 @bar(<4 x i32> %v, i32 %i) #0 {
entry:
  %cmp = icmp sgt i32 %i, 4
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %mul1 = mul nsw i32 %i, %i
  br label %return

if.else:                                          ; preds = %entry
  %add1 = add nsw i32 %i, %i
  %add2 = add nsw i32 %i, %i
  %add3 = add nsw i32 %i, %i
  %add4 = add nsw i32 %i, %i
  %add5 = add nsw i32 %i, %i
  %add6 = add nsw i32 %i, %i
  %vecext = extractelement <4 x i32> %v, i32 0
  %vecext7 = extractelement <4 x i32> %v, i32 1
  %add7 = add nsw i32 %vecext, %vecext7
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %mul1, %if.then ], [ %add7, %if.else ]
  ret i32 %retval.0
}

define i32 @foo(<4 x i32> %v, i32 %a) #1 {
; CHECK-LABEL: @foo(
; CHECK-NOT: call i32 @bar
; CHECK: ret
entry:
  %call = call i32 @bar(<4 x i32> %v, i32 %a)
  ret i32 %call
}

