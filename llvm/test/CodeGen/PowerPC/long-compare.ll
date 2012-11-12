; RUN: llc < %s -march=ppc32 | grep cntlzw 
; RUN: llc < %s -march=ppc32 | not grep xori 
; RUN: llc < %s -march=ppc32 | not grep "li "
; RUN: llc < %s -march=ppc32 | not grep "mr "

define i1 @test(i64 %x) {
  %tmp = icmp ult i64 %x, 4294967296
  ret i1 %tmp
}
