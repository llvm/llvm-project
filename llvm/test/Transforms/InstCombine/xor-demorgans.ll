; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {= or}
; PR3266
; XFAIL: *

define i1 @foo(i32 %x, i32 %y) nounwind {
.summary:
       %0 = icmp sgt i32 %x, 4         ; <i1> [#uses=1]
       %1 = icmp sgt i32 %y, 0         ; <i1> [#uses=1]
       %.demorgan = or i1 %1, %0               ; <i1> [#uses=1]
       %2 = xor i1 %.demorgan, true            ; <i1> [#uses=1]
       ret i1 %2
}
