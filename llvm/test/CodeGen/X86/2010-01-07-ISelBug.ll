; RUN: llc < %s -mtriple=x86_64-apple-darwin10
; rdar://r7519827

define i32 @t() nounwind ssp {
entry:
  br label %if.end.i11

if.end.i11:                                       ; preds = %lor.lhs.false.i10, %lor.lhs.false.i10, %lor.lhs.false.i10
  br i1 undef, label %for.body161, label %for.end197

for.body161:                                      ; preds = %if.end.i11
  br label %for.end197

for.end197:                                       ; preds = %for.body161, %if.end.i11
  %mlucEntry.4 = phi i96 [ undef, %for.body161 ], [ undef, %if.end.i11 ] ; <i96> [#uses=2]
  store i96 %mlucEntry.4, i96* undef, align 8
  %tmp172 = lshr i96 %mlucEntry.4, 64             ; <i96> [#uses=1]
  %tmp173 = trunc i96 %tmp172 to i32              ; <i32> [#uses=1]
  %tmp1.i1.i = call i32 @llvm.bswap.i32(i32 %tmp173) nounwind ; <i32> [#uses=1]
  store i32 %tmp1.i1.i, i32* undef, align 8
  unreachable

if.then283:                                       ; preds = %lor.lhs.false.i10, %do.end105, %for.end
  ret i32 undef
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone
