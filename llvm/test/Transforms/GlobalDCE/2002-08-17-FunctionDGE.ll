; Make sure that functions are removed successfully if they are referred to by
; a global that is dead.  Make sure any globals they refer to die as well.

; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; CHECK-NOT: foo
;; Unused, kills %foo
@b = internal global ptr @foo               ; <ptr> [#uses=0]

;; Should die when function %foo is killed
@foo.upgrd.1 = internal global i32 7            ; <ptr> [#uses=1]

 ;; dies when %b dies.
define internal i32 @foo() {
        %ret = load i32, ptr @foo.upgrd.1           ; <i32> [#uses=1]
        ret i32 %ret
}

