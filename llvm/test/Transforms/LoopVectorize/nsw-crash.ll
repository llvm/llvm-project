; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4

target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @test() {
entry:
  br i1 undef, label %while.end, label %while.body.lr.ph

while.body.lr.ph:
  br label %while.body

while.body:
  %it.sroa.0.091 = phi i32* [ undef, %while.body.lr.ph ], [ %incdec.ptr.i, %while.body ]
  %incdec.ptr.i = getelementptr inbounds i32, i32* %it.sroa.0.091, i64 1
  %inc32 = add i32 undef, 1                                        ; <------------- Make sure we don't set NSW flags to the undef.
  %cmp.i11 = icmp eq i32* %incdec.ptr.i, undef
  br i1 %cmp.i11, label %while.end, label %while.body

while.end:
  ret void
}


