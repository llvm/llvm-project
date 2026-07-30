; RUN: opt < %s -passes=instsimplify -S | grep 1065353216

define i32 @test() {
        %A = bitcast float 1.000000e+00 to i32          ; <i32> [#uses=1]
        ret i32 %A
}

