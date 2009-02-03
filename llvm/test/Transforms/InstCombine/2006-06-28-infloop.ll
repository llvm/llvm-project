; RUN: llvm-as < %s | opt -instcombine -disable-output

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8"

define void @test() {
entry:
        %tmp = getelementptr { i64, i64, i64, i64 }* null, i32 0, i32 3         ; <i64*> [#uses=1]
        %tmp.upgrd.1 = load i64* %tmp           ; <i64> [#uses=1]
        %tmp8.ui = load i64* null               ; <i64> [#uses=1]
        %tmp8 = bitcast i64 %tmp8.ui to i64             ; <i64> [#uses=1]
        %tmp9 = and i64 %tmp8, %tmp.upgrd.1             ; <i64> [#uses=1]
        %sext = trunc i64 %tmp9 to i32          ; <i32> [#uses=1]
        %tmp27.i = sext i32 %sext to i64                ; <i64> [#uses=1]
        tail call void @foo( i32 0, i64 %tmp27.i )
        unreachable
}

declare void @foo(i32, i64)

