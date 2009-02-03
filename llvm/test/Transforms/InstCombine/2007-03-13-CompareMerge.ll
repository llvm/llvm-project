; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {icmp sle}
; PR1244

define i1 @test(i32 %c.3.i, i32 %d.292.2.i) {
   %tmp266.i = icmp slt i32 %c.3.i, %d.292.2.i     
   %tmp276.i = icmp eq i32 %c.3.i, %d.292.2.i 
   %sel_tmp80 = or i1 %tmp266.i, %tmp276.i 
   ret i1 %sel_tmp80
}
