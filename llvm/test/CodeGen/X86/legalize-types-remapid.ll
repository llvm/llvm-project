; RUN: llc -mtriple=i386 -mcpu=generic -O0 -o /dev/null %s

@c = global i32 0
@d = global <2 x i64> zeroinitializer

define void @test() {
bb1:
  %t0 = load <2 x i64>, <2 x i64>* @d
  %t0.i0 = extractelement <2 x i64> %t0, i32 0
  %t0.i0.cast = bitcast i64 %t0.i0 to <2 x i32>
  %t0.i0.cast.i0 = extractelement <2 x i32> %t0.i0.cast, i32 0
  store volatile i32 %t0.i0.cast.i0, i32* @c
  %t0.i0.cast.i1 = extractelement <2 x i32> %t0.i0.cast, i32 1
  store volatile i32 %t0.i0.cast.i1, i32* @c
  ret void
}
