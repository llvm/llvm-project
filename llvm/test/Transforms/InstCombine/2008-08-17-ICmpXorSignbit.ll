; RUN: opt < %s -instcombine -S | not grep xor

define i1 @test1(i8 %x, i8 %y) {
  %X = xor i8 %x, 128
  %Y = xor i8 %y, 128
  %tmp = icmp slt i8 %X, %Y
  ret i1 %tmp
}

define i1 @test2(i8 %x, i8 %y) {
  %X = xor i8 %x, 128
  %Y = xor i8 %y, 128
  %tmp = icmp ult i8 %X, %Y
  ret i1 %tmp
}

define i1 @test3(i8 %x) {
  %X = xor i8 %x, 128
  %tmp = icmp uge i8 %X, 15
  ret i1 %tmp
}

define i1 @test4(i8 %x, i8 %y) {
  %X = xor i8 %x, 127
  %Y = xor i8 %y, 127
  %tmp = icmp slt i8 %X, %Y
  ret i1 %tmp
}

define i1 @test5(i8 %x, i8 %y) {
  %X = xor i8 %x, 127
  %Y = xor i8 %y, 127
  %tmp = icmp ult i8 %X, %Y
  ret i1 %tmp
}

define i1 @test6(i8 %x) {
  %X = xor i8 %x, 127
  %tmp = icmp uge i8 %X, 15
  ret i1 %tmp
}
