; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
; PR2733

%t1 = type i32
%t2 = type { %t1 }
@i1 = constant %t2 { %t1 15 } 
