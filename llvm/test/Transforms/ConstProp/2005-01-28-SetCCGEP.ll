; RUN: opt < %s -constprop -S | \
; RUN:    not grep "ret i1 false"

@b = external global [2 x {  }]         ; <[2 x {  }]*> [#uses=2]

define i1 @f() {
        %tmp.2 = icmp eq {  }* getelementptr ([2 x {  }], [2 x {  }]* @b, i32 0, i32 0), getelementptr ([2 x {  }], [2 x {  }]* @b, i32 0, i32 1)                ; <i1> [#uses=1]
        ret i1 %tmp.2
}

