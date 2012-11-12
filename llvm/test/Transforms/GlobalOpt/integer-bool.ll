; RUN: opt < %s -globalopt -instcombine | \
; RUN:    llvm-dis | grep "ret i1 true"

;; check that global opt turns integers that only hold 0 or 1 into bools.

@G = internal global i32 0              ; <i32*> [#uses=3]

define void @set1() {
        store i32 0, i32* @G
        ret void
}

define void @set2() {
        store i32 1, i32* @G
        ret void
}

define i1 @get() {
        %A = load i32* @G               ; <i32> [#uses=1]
        %C = icmp slt i32 %A, 2         ; <i1> [#uses=1]
        ret i1 %C
}

