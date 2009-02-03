; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep {icmp ne\|icmp ule\|icmp uge}

define i32 @test1(i32 %X, i32 %Y) {
        %C = icmp ne i32 %X, %Y         ; <i1> [#uses=1]
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 12

F:              ; preds = %0
        ret i32 123
}

define i32 @test2(i32 %X, i32 %Y) {
        %C = icmp ule i32 %X, %Y                ; <i1> [#uses=1]
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 12

F:              ; preds = %0
        ret i32 123
}

define i32 @test3(i32 %X, i32 %Y) {
        %C = icmp uge i32 %X, %Y                ; <i1> [#uses=1]
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 12

F:              ; preds = %0
        ret i32 123
}

