; RUN: opt < %s -passes=instsimplify -S | \
; RUN:   grep "i32 -1"
; RUN: opt < %s -passes=instsimplify -S | \
; RUN:   not grep zeroinitializer

define <4 x i32> @test() {
        %tmp40 = bitcast <2 x i64> bitcast (<4 x i32> < i32 0, i32 0, i32 -1, i32 0 > to <2 x i64>) to <4 x i32>; <<4 x i32>> [#uses=1]
        ret <4 x i32> %tmp40
}

