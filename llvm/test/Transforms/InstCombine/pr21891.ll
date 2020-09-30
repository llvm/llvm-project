; RUN: opt < %s -instcombine

define i32 @f(i32 %theNumber) {
entry:
  %cmp = icmp sgt i32 %theNumber, -1
  call void @llvm.assume(i1 %cmp)
  br i1 true, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %shl = shl nuw i32 %theNumber, 1
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %phi = phi i32 [ %shl, %if.then ], [ undef, %entry ]
  ret i32 %phi
}

declare void @llvm.assume(i1)
