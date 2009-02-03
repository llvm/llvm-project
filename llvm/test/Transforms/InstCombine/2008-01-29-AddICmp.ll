; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {a.off}
; PR1949

define i1 @test1(i32 %a) {
  %a.off = add i32 %a, 4          ; <i32> [#uses=1]
  %C = icmp ult i32 %a.off, 4             ; <i1> [#uses=1]
  ret i1 %C
}

define i1 @test2(i32 %a) {
  %a.off = sub i32 %a, 4          ; <i32> [#uses=1]
  %C = icmp ugt i32 %a.off, -5             ; <i1> [#uses=1]
  ret i1 %C
}

define i1 @test3(i32 %a) {
  %a.off = add i32 %a, 4          ; <i32> [#uses=1]
  %C = icmp slt i32 %a.off, 2147483652             ; <i1> [#uses=1]
  ret i1 %C
}
