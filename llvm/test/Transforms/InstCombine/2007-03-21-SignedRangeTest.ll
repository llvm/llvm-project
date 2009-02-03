; For PR1248
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | grep {ugt i32 .*, 11}
define i1 @test(i32 %tmp6) {
  %tmp7 = sdiv i32 %tmp6, 12     ; <i32> [#uses=1]
  icmp ne i32 %tmp7, -6           ; <i1>:1 [#uses=1]
  ret i1 %1
}
