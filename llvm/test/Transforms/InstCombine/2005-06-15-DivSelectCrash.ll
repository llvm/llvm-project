; RUN: opt < %s -passes=instcombine -disable-output

define i32 @_Z13func_31585107li(i32 %l_39521025, i32 %l_59244666) {
        %shortcirc_val = select i1 false, i32 1, i32 0          ; <i32> [#uses=1]
        %tmp.8 = udiv i32 0, %shortcirc_val             ; <i32> [#uses=1]
        %tmp.9 = icmp eq i32 %tmp.8, 0          ; <i1> [#uses=1]
        %retval = select i1 %tmp.9, i32 %l_59244666, i32 -1621308501            ; <i32> [#uses=1]
        ret i32 %retval
}

