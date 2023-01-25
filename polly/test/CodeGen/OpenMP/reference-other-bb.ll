; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR

; IR: @foo_polly_subfn
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i32 %sendcount, ptr %recvbuf) {
entry:
  br label %sw.bb3

sw.bb3:
  %cmp75 = icmp sgt i32 %sendcount, 0
  br i1 %cmp75, label %for.body, label %end

for.body:
  %i.16 = phi i32 [ %inc04, %for.body ], [ 0, %sw.bb3 ]
  %idxprom11 = sext i32 %i.16 to i64
  %arrayidx12 = getelementptr inbounds double, ptr %recvbuf, i64 %idxprom11
  store double 1.0, ptr %arrayidx12, align 8
  %inc04 = add nsw i32 %i.16, 1
  %cmp7 = icmp slt i32 %inc04, %sendcount
  br i1 %cmp7, label %for.body, label %end

end:
  ret void
}
