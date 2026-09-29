; RUN: llc < %s -mtriple=i686--

define i32 @test(i16 %tmp40414244) {
  %tmp48 = call i32 asm sideeffect "inl ${1:w}, $0", "={ax},N{dx},~{dirflag},~{fpsr},~{flags}"( i16 %tmp40414244 )
  ret i32 %tmp48
}

define i32 @test2(i16 %tmp40414244) {
  %tmp48 = call i32 asm sideeffect "inl ${1:w}, $0", "={ax},N{dx},~{dirflag},~{fpsr},~{flags}"( i16 14 )
  ret i32 %tmp48
}
