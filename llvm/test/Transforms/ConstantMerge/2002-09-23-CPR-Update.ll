; RUN: opt < %s -passes=constmerge > /dev/null

@foo.upgrd.1 = internal constant { i32 } { i32 7 }              ; <ptr> [#uses=1]
@bar = internal constant { i32 } { i32 7 }              ; <ptr> [#uses=1]

declare i32 @test(ptr)

define void @foo() {
        call i32 @test( ptr @foo.upgrd.1 )              ; <i32>:1 [#uses=0]
        call i32 @test( ptr @bar )              ; <i32>:2 [#uses=0]
        ret void
}

