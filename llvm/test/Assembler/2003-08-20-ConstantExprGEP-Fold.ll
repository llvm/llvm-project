; RUN: llvm-as < %s | opt -instcombine -simplifycfg | llvm-dis | not grep br

@.str_1 = internal constant [6 x i8] c"_Bool\00"                ; <[6 x i8]*> [#uses=2]

define i32 @test() {
        %tmp.54 = load i8* getelementptr ([6 x i8]* @.str_1, i64 0, i64 1)            ; <i8> [#uses=1]
        %tmp.55 = icmp ne i8 %tmp.54, 66                ; <i1> [#uses=1]
        br i1 %tmp.55, label %then.7, label %endif.7

then.7:         ; preds = %then.7, %0
        br label %then.7

endif.7:                ; preds = %0
        ret i32 0
}

