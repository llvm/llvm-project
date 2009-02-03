; RUN: llvm-as < %s | llc -march=arm | \
; RUN:   grep {ldr r0} | count 3

define i32 @f1(i32* %v) {
entry:
        %tmp = load i32* %v             ; <i32> [#uses=1]
        ret i32 %tmp
}

define i32 @f2(i32* %v) {
entry:
        %tmp2 = getelementptr i32* %v, i32 1023         ; <i32*> [#uses=1]
        %tmp = load i32* %tmp2          ; <i32> [#uses=1]
        ret i32 %tmp
}

define i32 @f3(i32* %v) {
entry:
        %tmp2 = getelementptr i32* %v, i32 1024         ; <i32*> [#uses=1]
        %tmp = load i32* %tmp2          ; <i32> [#uses=1]
        ret i32 %tmp
}

