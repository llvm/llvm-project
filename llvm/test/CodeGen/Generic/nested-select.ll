; RUN: llc < %s -o /dev/null

; Test that select of a select works

%typedef.tree = type opaque

define i32 @ic_test(double %p.0.2.0.val, double %p.0.2.1.val, double %p.0.2.2.val, ptr %t) {
        %result.1.0 = zext i1 false to i32              ; <i32> [#uses=1]
        %tmp.55 = fcmp oge double 0.000000e+00, 1.000000e+00            ; <i1> [#uses=1]
        %tmp.66 = fdiv double 0.000000e+00, 0.000000e+00                ; <double> [#uses=1]
        br label %N

N:              ; preds = %0
        %result.1.1 = select i1 %tmp.55, i32 0, i32 %result.1.0         ; <i32> [#uses=1]
        %tmp.75 = fcmp oge double %tmp.66, 1.000000e+00         ; <i1> [#uses=1]
        %retval1 = select i1 %tmp.75, i32 0, i32 %result.1.1            ; <i32> [#uses=1]
        ret i32 %retval1
}

