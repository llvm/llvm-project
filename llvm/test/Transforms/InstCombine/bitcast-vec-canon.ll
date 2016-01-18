; RUN: opt < %s -instcombine -S | grep element | count 4

define double @a(<1 x i64> %y) {
  %c = bitcast <1 x i64> %y to double
  ret double %c 
}

define i64 @b(<1 x i64> %y) {
  %c = bitcast <1 x i64> %y to i64
  ret i64 %c 
}

define <1 x i64> @c(double %y) {
  %c = bitcast double %y to <1 x i64>
  ret <1 x i64> %c
}

define <1 x i64> @d(i64 %y) {
  %c = bitcast i64 %y to <1 x i64>
  ret <1 x i64> %c
}

