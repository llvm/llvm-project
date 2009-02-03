; RUN: llvm-as %s -f -o %t.bc
; RUN: lli -debug-only=jit %t.bc |& not grep {Finished CodeGen of .*Function: F}
@.str_1 = internal constant [7 x i8] c"IN F!\0A\00"             ; <[7 x i8]*> [#uses=1]
@.str_2 = internal constant [7 x i8] c"IN G!\0A\00"             ; <[7 x i8]*> [#uses=1]
@Ptrs = internal constant [2 x void (...)*] [ void (...)* bitcast (void ()* @F to void (...)*), void (...)* bitcast (void ()* @G to void (...)*) ]           ; <[2 x void (...)*]*> [#uses=1]

declare i32 @printf(i8*, ...)

define internal void @F() {
entry:
        %tmp.0 = call i32 (i8*, ...)* @printf( i8* getelementptr ([7 x i8]* @.str_1, i32 0, i32 0) )            ; <i32> [#uses=0]
        ret void
}

define internal void @G() {
entry:
        %tmp.0 = call i32 (i8*, ...)* @printf( i8* getelementptr ([7 x i8]* @.str_2, i32 0, i32 0) )            ; <i32> [#uses=0]
        ret void
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
        %tmp.3 = and i32 %argc, 1               ; <i32> [#uses=1]
        %tmp.4 = getelementptr [2 x void (...)*]* @Ptrs, i32 0, i32 %tmp.3              ; <void (...)**> [#uses=1]
        %tmp.5 = load void (...)** %tmp.4               ; <void (...)*> [#uses=1]
        %tmp.5_c = bitcast void (...)* %tmp.5 to void ()*               ; <void ()*> [#uses=1]
        call void %tmp.5_c( )
        ret i32 0
}
