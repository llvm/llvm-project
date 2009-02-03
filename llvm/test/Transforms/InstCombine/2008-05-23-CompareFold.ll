; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {ret i1 false}
; PR2359
define i1 @f(i8* %x) {
entry:
       %tmp462 = load i8* %x, align 1          ; <i8> [#uses=1]
       %tmp462463 = sitofp i8 %tmp462 to float         ; <float> [#uses=1]
       %tmp464 = fcmp ugt float %tmp462463, 0x47EFFFFFE0000000         ; <i1>
       ret i1 %tmp464
}


