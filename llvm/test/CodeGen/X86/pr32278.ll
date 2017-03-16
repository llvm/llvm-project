; PR32278

; RUN: llc -mtriple=x86_64-unknown < %s

define i8 @foo_v4i1_0_0_1_1_2_2_3_3(i8 %in) {
  %trunc = trunc i8 %in to i4
  %mask = bitcast i4 %trunc to <4 x i1>
  %s = shufflevector <4 x i1> %mask, <4 x i1> undef, <8 x i32> <i32 0, i32 0, i32 1, i32 1, i32 2, i32 2, i32 3, i32 3>
  %b = bitcast <8 x i1> %s to i8
  ret i8 %b
}
