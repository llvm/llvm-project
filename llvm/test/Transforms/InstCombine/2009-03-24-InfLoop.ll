; PR3874
; RUN: opt < %s -instcombine | llvm-dis
  define i1 @test(i32 %x) {
    %A = lshr i32 3968, %x
    %B = and i32 %A, 1
    %C = icmp eq i32 %B, 0
    ret i1 %C
  }

