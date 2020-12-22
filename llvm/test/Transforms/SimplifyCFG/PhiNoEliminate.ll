; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | \
; RUN:   not grep select

;; The PHI node in this example should not be turned into a select, as we are
;; not able to ifcvt the entire block.  As such, converting to a select just
;; introduces inefficiency without saving copies.

define i32 @bar(i1 %C) {
entry:
        br i1 %C, label %then, label %endif
then:           ; preds = %entry
        %tmp.3 = call i32 @qux( )               ; <i32> [#uses=0]
        br label %endif
endif:          ; preds = %then, %entry
        %R = phi i32 [ 123, %entry ], [ 12312, %then ]          ; <i32> [#uses=1]
        ;; stuff to disable tail duplication
        call i32 @qux( )                ; <i32>:0 [#uses=0]
        call i32 @qux( )                ; <i32>:1 [#uses=0]
        call i32 @qux( )                ; <i32>:2 [#uses=0]
        call i32 @qux( )                ; <i32>:3 [#uses=0]
        call i32 @qux( )                ; <i32>:4 [#uses=0]
        call i32 @qux( )                ; <i32>:5 [#uses=0]
        call i32 @qux( )                ; <i32>:6 [#uses=0]
        ret i32 %R
}

declare i32 @qux()
