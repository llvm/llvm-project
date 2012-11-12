; RUN: opt < %s -instcombine -S | grep "add" | count 1

define i32 @foo(i32 %a) {
entry:
        %tmp15 = sub i32 99, %a         ; <i32> [#uses=2]
        %tmp16 = icmp slt i32 %tmp15, 0         ; <i1> [#uses=1]
        %smax = select i1 %tmp16, i32 0, i32 %tmp15             ; <i32> [#uses=1]
        %tmp12 = add i32 %smax, %a              ; <i32> [#uses=1]
        %tmp13 = add i32 %tmp12, 1              ; <i32> [#uses=1]
        ret i32 %tmp13
}

define i32 @bar(i32 %a) {
entry:
        %tmp15 = sub i32 99, %a         ; <i32> [#uses=2]
        %tmp16 = icmp slt i32 %tmp15, 0         ; <i1> [#uses=1]
        %smax = select i1 %tmp16, i32 0, i32 %tmp15             ; <i32> [#uses=1]
        %tmp12 = add i32 %smax, %a              ; <i32> [#uses=1]
        ret i32 %tmp12
}

define i32 @fun(i32 %a) {
entry:
        %tmp15 = sub i32 99, %a         ; <i32> [#uses=1]
        %tmp16 = icmp slt i32 %a, 0         ; <i1> [#uses=1]
        %smax = select i1 %tmp16, i32 0, i32 %tmp15             ; <i32> [#uses=1]
        %tmp12 = add i32 %smax, %a              ; <i32> [#uses=1]
        ret i32 %tmp12
}
