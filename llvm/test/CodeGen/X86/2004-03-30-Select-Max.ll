; RUN: llvm-as < %s | llc -march=x86 | not grep {j\[lgbe\]}

define i32 @max(i32 %A, i32 %B) {
        %gt = icmp sgt i32 %A, %B               ; <i1> [#uses=1]
        %R = select i1 %gt, i32 %A, i32 %B              ; <i32> [#uses=1]
        ret i32 %R
}

