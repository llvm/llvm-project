; RUN: opt -S -sccp %s

@A = external constant i32

define void @test1() {
BB4:
  %A20 = alloca i1
  %A15 = alloca i64
  %A7 = alloca i64
  %A3 = alloca i32**
  %P = getelementptr i32, i32* @A, i32 0
  %B = ptrtoint i32* %P to i64
  %B8 = shl i64 %B, 9223372036854775807
  %G10 = getelementptr i32*, i32** undef, i64 %B
  %B10 = urem i64 %B, %B8
  %B12 = shl i64 %B, %B
  %BB = and i64 %B, %B8
  %B1 = xor i64 %B, %B
  %B23 = lshr i64 %B8, undef
  %C5 = icmp uge i64 %B, %B10
  %C17 = fcmp ord double 4.940660e-324, 0x7FEFFFFFFFFFFFFF
  %C2 = icmp uge i1 %C17, false
  %G = getelementptr i32, i32* %P, i1 %C17
  %X = select i1 false, i712 0, i712 1
  %C4 = icmp ule i1 true, false
  %B3 = xor i1 %C17, %C2
  %C33 = icmp slt i1 false, %C5
  %B15 = sub i64 %B8, %B23
  %C18 = icmp slt i64 undef, %BB
  %G29 = getelementptr i32**, i32*** undef, i64 %B15
  %C35 = icmp eq i1 %C17, undef
  %C31 = icmp ult i1 %C35, %C5
  %C29 = icmp sle i1 true, %C5
  %C16 = icmp ne i16 -1, -32768
  %A24 = alloca i1
  %A21 = alloca i1
  %A25 = alloca i32**
  %C7 = icmp ule i1 %C4, %B3
  %C14 = icmp slt i64 %B8, 0
  ret void
}
