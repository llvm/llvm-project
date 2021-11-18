; RUN: opt < %s -passes=constmerge > /dev/null

@foo.upgrd.1 = internal constant { i32 } { i32 7 }              ; <{ i32 }*> [#uses=1]
@bar = internal constant { i32 } { i32 7 }              ; <{ i32 }*> [#uses=1]

declare i32 @test(i32*)

define void @foo() {
        call i32 @test( i32* getelementptr ({ i32 }, { i32 }* @foo.upgrd.1, i64 0, i32 0) )              ; <i32>:1 [#uses=0]
        call i32 @test( i32* getelementptr ({ i32 }, { i32 }* @bar, i64 0, i32 0) )              ; <i32>:2 [#uses=0]
        ret void
}

