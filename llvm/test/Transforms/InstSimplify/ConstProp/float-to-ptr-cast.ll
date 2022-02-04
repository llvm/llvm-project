; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

define i32* @test1() {
        %X = inttoptr i64 0 to i32*             ; <i32*> [#uses=1]
        ret i32* %X
}

; CHECK:  ret i32* null

define i32* @test2() {
        ret i32* null
}

; CHECK:  ret i32* null

