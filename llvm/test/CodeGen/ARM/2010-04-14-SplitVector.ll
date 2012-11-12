; RUN: llc < %s -march=arm -mcpu=arm1136jf-s
; Radar 7854640

define void @test() nounwind {
bb:
  br i1 undef, label %bb9, label %bb10

bb9:
  %tmp63 = bitcast <4 x float> zeroinitializer to i128
  %tmp64 = trunc i128 %tmp63 to i32
  br label %bb10

bb10:
  %0 = phi i32 [ %tmp64, %bb9 ], [ undef, %bb ]
  ret void
}
