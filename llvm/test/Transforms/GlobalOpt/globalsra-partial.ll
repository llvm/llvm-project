; In this case, the global can only be broken up by one level.

; RUN: llvm-as < %s | opt -globalopt | llvm-dis | not grep 12345

@G = internal global { i32, [4 x float] } zeroinitializer               ; <{ i32, [4 x float] }*> [#uses=3]

define void @onlystore() {
        store i32 12345, i32* getelementptr ({ i32, [4 x float] }* @G, i32 0, i32 0)
        ret void
}

define void @storeinit(i32 %i) {
        %Ptr = getelementptr { i32, [4 x float] }* @G, i32 0, i32 1, i32 %i             ; <float*> [#uses=1]
        store float 1.000000e+00, float* %Ptr
        ret void
}

define float @readval(i32 %i) {
        %Ptr = getelementptr { i32, [4 x float] }* @G, i32 0, i32 1, i32 %i             ; <float*> [#uses=1]
        %V = load float* %Ptr           ; <float> [#uses=1]
        ret float %V
}

