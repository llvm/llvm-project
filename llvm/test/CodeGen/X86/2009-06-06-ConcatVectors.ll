; RUN: llc < %s

define <2 x i64> @_mm_movpi64_pi64(<1 x i64> %a, <1 x i64> %b) nounwind readnone {
entry:
  %0 = shufflevector <1 x i64> %a, <1 x i64> %b, <2 x i32> <i32 0, i32 1>
	ret <2 x i64> %0
}

