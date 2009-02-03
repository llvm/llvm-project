; RUN: llvm-as < %s | llc -march=x86 | not grep add

define i1 @X(i32 %X) {
        %Y = add i32 %X, 14             ; <i32> [#uses=1]
        %Z = icmp ne i32 %Y, 12345              ; <i1> [#uses=1]
        ret i1 %Z
}

