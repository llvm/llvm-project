; RUN: opt < %s -globaldce

;; Should die when function %foo is killed
@foo.upgrd.1 = internal global i32 7            ; <i32*> [#uses=3]
@bar = internal global [2 x { i32*, i32 }] [ { i32*, i32 } { i32* @foo.upgrd.1, i32 7 }, { i32*, i32 } { i32* @foo.upgrd.1, i32 1 } ]            ; <[2 x { i32*, i32 }]*> [#uses=0]

define internal i32 @foo() {
        %ret = load i32, i32* @foo.upgrd.1           ; <i32> [#uses=1]
        ret i32 %ret
}

