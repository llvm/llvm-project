; RUN: opt < %s -passes=instsimplify -S | \
; RUN:    not grep "ret i1 false"

@b = external global [2 x {  }]         ; <ptr> [#uses=2]

define i1 @f() {
        %tmp.2 = icmp eq ptr @b, getelementptr ([2 x {  }], ptr @b, i32 0, i32 1)                ; <i1> [#uses=1]
        ret i1 %tmp.2
}

