; RUN: llvm-as < %s | llc -march=ppc32 

define i32 @main() {
        %setle = icmp sle i64 1, 0              ; <i1> [#uses=1]
        %select = select i1 true, i1 %setle, i1 true            ; <i1> [#uses=0]
        ret i32 0
}

