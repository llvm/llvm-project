; RUN: llc < %s -mtriple=i686-- | grep movl | count 1

@dst = global i32 0             ; <i32*> [#uses=1]
@ptr = global i32* null         ; <i32**> [#uses=1]

define void @test() {
        store i32* @dst, i32** @ptr
        ret void
}

