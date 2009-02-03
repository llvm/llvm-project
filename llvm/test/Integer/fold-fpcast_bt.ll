; RUN: llvm-as < %s | llvm-dis | not grep bitcast

define i60 @test1() {
   ret i60 fptoui(float 0x400D9999A0000000 to i60)
}

define float @test2() {
  ret float uitofp(i60 17 to float)
}

define i64 @test3() {
  ret i64 bitcast (double 0x400921FB4D12D84A to i64)
}

define double @test4() {
  ret double bitcast (i64 42 to double)
}

define i30 @test5() {
  ret i30 fptoui(float 0x400D9999A0000000 to i30)
}

define float @test6() {
  ret float uitofp(i30 17 to float)
}

define i64 @test7() {
  ret i64 bitcast (double 0x400921FB4D12D84A to i64)
}

define double @test8() {
  ret double bitcast (i64 42 to double)
}
