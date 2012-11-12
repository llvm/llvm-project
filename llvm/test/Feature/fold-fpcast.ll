; RUN: llvm-as < %s | llvm-dis | not grep bitcast

define i32 @test1() {
   ret i32 bitcast(float 0x400D9999A0000000 to i32)
}

define float @test2() {
  ret float bitcast(i32 17 to float)
}

define i64 @test3() {
  ret i64 bitcast (double 0x400921FB4D12D84A to i64)
}

define double @test4() {
  ret double bitcast (i64 42 to double)
}

