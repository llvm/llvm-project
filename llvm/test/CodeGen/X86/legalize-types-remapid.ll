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

define void @PR45049() local_unnamed_addr {
so_basic:
  %a0 = load i1, i1* undef, align 1
  %a1 = select i1 %a0, i542 4374501449566023848745004454235242730706338861786424872851541212819905998398751846447026354046107648, i542 0 ; constant is: i542 1 << 331
  %a00 = zext i1 %a0 to i542
  %a11 = shl i542 %a00, 331
  %a2 = shl i542 %a00, 330
  %a4 = or i542 %a1, %a2
  %a05 = zext i1 %a0 to i488
  %a55 = shl i488 %a05, 111
  store i542 %a4, i542* undef, align 8
  store i488 %a55, i488* undef, align 8
  ret void
}
