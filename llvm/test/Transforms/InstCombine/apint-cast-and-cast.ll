; RUN: opt < %s -instcombine -S | not grep bitcast

define i19 @test1(i43 %val) {
  %t1 = bitcast i43 %val to i43 
  %t2 = and i43 %t1, 1
  %t3 = trunc i43 %t2 to i19
  ret i19 %t3
}

define i73 @test2(i677 %val) {
  %t1 = bitcast i677 %val to i677 
  %t2 = and i677 %t1, 1
  %t3 = trunc i677 %t2 to i73
  ret i73 %t3
}
