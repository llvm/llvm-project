; RUN: llc < %s -mcpu=generic -march=x86 -mattr=+sse2 -mtriple=i686-apple-darwin8 | \
; RUN:   grep "subl.*60"
; RUN: llc < %s -mcpu=generic -march=x86 -mattr=+sse2 -mtriple=i686-apple-darwin8 | \
; RUN:   grep "movaps.*32"


define void @test() {
        tail call void @xx( i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, <2 x i64> bitcast (<4 x i32> < i32 4, i32 3, i32 2, i32 1 > to <2 x i64>), <2 x i64> bitcast (<4 x i32> < i32 8, i32 7, i32 6, i32 5 > to <2 x i64>), <2 x i64> bitcast (<4 x i32> < i32 6, i32 4, i32 2, i32 0 > to <2 x i64>), <2 x i64> bitcast (<4 x i32> < i32 8, i32 4, i32 2, i32 1 > to <2 x i64>), <2 x i64> bitcast (<4 x i32> < i32 0, i32 1, i32 3, i32 9 > to <2 x i64>) )
        ret void
}

declare void @xx(i32, i32, i32, i32, i32, i32, i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>)

