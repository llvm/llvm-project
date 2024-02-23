; REQUIRES: asserts
; RUN: not --crash opt -mtriple=x86_64 -mattr=-avx,-avx2,-avx512f,+sse,-sse2,-sse3,-sse4.2 -passes=loop-vectorize -S < %s
; RUN: not --crash opt -mtriple=x86_64 -mattr=-avx,-avx2,-avx512f,+sse,-sse2,-sse3,-sse4.2 -passes=loop-vectorize -force-vector-width=4 -S < %s

@h = global i64 0

define void @test(ptr %p) {
entry:
  br label %for.body

for.body:
  %idx.ext.merge = phi i64 [ 1, %entry ], [ %idx, %for.body ]
  %inc.merge = phi i16 [ 1, %entry ], [ %inc, %for.body ]
  %idx.merge = phi i64 [ 0, %entry ], [ %idx.ext.merge, %for.body ]
  %add = shl i64 %idx.merge, 1
  %arrayidx = getelementptr i64, ptr %p, i64 %add
  store i64 0, ptr %arrayidx
  %inc = add i16 %inc.merge, 1
  %idx = zext i16 %inc to i64
  %gep = getelementptr i64, ptr %p, i64 %idx
  %cmp = icmp ugt ptr %gep, @h
  br i1 %cmp, label %exit, label %for.body

exit:
  ret void
}
