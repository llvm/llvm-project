; RUN: llc < %s

define float @t(i64 %u_arg) {
        %u = bitcast i64 %u_arg to i64          ; <i64> [#uses=1]
        %tmp5 = add i64 %u, 9007199254740991            ; <i64> [#uses=1]
        %tmp = icmp ugt i64 %tmp5, 18014398509481982            ; <i1> [#uses=1]
        br i1 %tmp, label %T, label %F

T:              ; preds = %0
        ret float 1.000000e+00

F:              ; preds = %0
        call float @t( i64 0 )          ; <float>:1 [#uses=0]
        ret float 0.000000e+00
}

