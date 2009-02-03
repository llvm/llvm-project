; RUN: llvm-as < %s | llc -march=x86-64  -tailcallopt  | grep TAILCALL
; Expect 2 rep;movs because of tail call byval lowering.
; RUN: llvm-as < %s | llc -march=x86-64  -tailcallopt  | grep rep | wc -l | grep 2
; A sequence of copyto/copyfrom virtual registers is used to deal with byval
; lowering appearing after moving arguments to registers. The following two
; checks verify that the register allocator changes those sequences to direct
; moves to argument register where it can (for registers that are not used in 
; byval lowering - not rsi, not rdi, not rcx).
; Expect argument 4 to be moved directly to register edx.
; RUN: llvm-as < %s | llc -march=x86-64  -tailcallopt  | grep movl | grep {7} | grep edx
; Expect argument 6 to be moved directly to register r8.
; RUN: llvm-as < %s | llc -march=x86-64  -tailcallopt  | grep movl | grep {17} | grep r8

%struct.s = type { i64, i64, i64, i64, i64, i64, i64, i64,
                   i64, i64, i64, i64, i64, i64, i64, i64,
                   i64, i64, i64, i64, i64, i64, i64, i64 }

declare  fastcc i64 @tailcallee(%struct.s* byval %a, i64 %val, i64 %val2, i64 %val3, i64 %val4, i64 %val5)


define  fastcc i64 @tailcaller(i64 %b, %struct.s* byval %a) {
entry:
        %tmp2 = getelementptr %struct.s* %a, i32 0, i32 1
        %tmp3 = load i64* %tmp2, align 8
        %tmp4 = tail call fastcc i64 @tailcallee(%struct.s* %a byval, i64 %tmp3, i64 %b, i64 7, i64 13, i64 17)
        ret i64 %tmp4
}


