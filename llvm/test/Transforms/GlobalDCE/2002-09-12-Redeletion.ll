; RUN: opt < %s -passes=globaldce

;; Should die when function %foo is killed
@foo.upgrd.1 = internal global i32 7            ; <ptr> [#uses=3]
@bar = internal global [2 x { ptr, i32 }] [ { ptr, i32 } { ptr @foo.upgrd.1, i32 7 }, { ptr, i32 } { ptr @foo.upgrd.1, i32 1 } ]            ; <ptr> [#uses=0]

define internal i32 @foo() {
        %ret = load i32, ptr @foo.upgrd.1           ; <i32> [#uses=1]
        ret i32 %ret
}

