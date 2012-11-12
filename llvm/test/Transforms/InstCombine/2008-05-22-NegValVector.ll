; RUN: opt < %s -instcombine -S | not grep sub

define <3 x i8> @f(<3 x i8> %a) {
  %A = sub <3 x i8> zeroinitializer, %a
  %B = mul <3 x i8> %A, <i8 5, i8 5, i8 5>
  ret <3 x i8> %B
}

