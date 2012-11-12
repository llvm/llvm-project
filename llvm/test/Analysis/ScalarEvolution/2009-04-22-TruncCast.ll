; RUN: opt < %s -analyze -scalar-evolution | grep "(trunc i" | not grep ext

define i16 @test1(i8 %x) {
  %A = sext i8 %x to i32
  %B = trunc i32 %A to i16
  ret i16 %B
}

define i8 @test2(i16 %x) {
  %A = sext i16 %x to i32
  %B = trunc i32 %A to i8
  ret i8 %B
}

define i16 @test3(i16 %x) {
  %A = sext i16 %x to i32
  %B = trunc i32 %A to i16
  ret i16 %B
}

define i16 @test4(i8 %x) {
  %A = zext i8 %x to i32
  %B = trunc i32 %A to i16
  ret i16 %B
}

define i8 @test5(i16 %x) {
  %A = zext i16 %x to i32
  %B = trunc i32 %A to i8
  ret i8 %B
}

define i16 @test6(i16 %x) {
  %A = zext i16 %x to i32
  %B = trunc i32 %A to i16
  ret i16 %B
}
