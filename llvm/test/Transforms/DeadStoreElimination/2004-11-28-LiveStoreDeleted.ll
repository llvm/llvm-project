; RUN: llvm-as < %s | opt -dse -scalarrepl -instcombine | \
; RUN:   llvm-dis | not grep {ret i32 undef}

define i32 @test(double %__x) {
        %__u = alloca { [3 x i32] }             ; <{ [3 x i32] }*> [#uses=2]
        %tmp.1 = bitcast { [3 x i32] }* %__u to double*         ; <double*> [#uses=1]
        store double %__x, double* %tmp.1
        %tmp.4 = getelementptr { [3 x i32] }* %__u, i32 0, i32 0, i32 1         ; <i32*> [#uses=1]
        %tmp.5 = load i32* %tmp.4               ; <i32> [#uses=1]
        %tmp.6 = icmp slt i32 %tmp.5, 0         ; <i1> [#uses=1]
        %tmp.7 = zext i1 %tmp.6 to i32          ; <i32> [#uses=1]
        ret i32 %tmp.7
}

