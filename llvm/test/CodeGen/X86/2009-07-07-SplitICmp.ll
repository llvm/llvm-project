; RUN: llc < %s -mtriple=i686--

define void @test2(<2 x i32> %A, <2 x i32> %B, ptr %C) nounwind {
       %D = icmp sgt <2 x i32> %A, %B
       %E = zext <2 x i1> %D to <2 x i32>
       store <2 x i32> %E, ptr %C
       ret void
}
