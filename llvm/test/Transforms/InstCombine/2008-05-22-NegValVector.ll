; RUN: opt < %s -instcombine -S | not grep sub

define <3 x i8> @f(<3 x i8> %a) {
  %A = sub <3 x i8> zeroinitializer, %a
  %B = mul <3 x i8> %A, <i8 5, i8 5, i8 5>
  ret <3 x i8> %B
}

define <3 x i4> @g(<3 x i4> %a) {
  %A = sub <3 x i4> zeroinitializer, %a
  %B = mul <3 x i4> %A, <i4 5, i4 5, i4 5>
  ret <3 x i4> %B
}

