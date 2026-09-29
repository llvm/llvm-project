; Double return in abicall (default)
; RUN: llc < %s -mtriple=mips
; PR2615

define double @main(...) {
entry:
        %retval = alloca double         ; <ptr> [#uses=3]
        store double 0.000000e+00, ptr %retval
        %r = alloca double              ; <ptr> [#uses=1]
        load double, ptr %r         ; <double>:0 [#uses=1]
        store double %0, ptr %retval
        br label %return

return:         ; preds = %entry
        load double, ptr %retval            ; <double>:1 [#uses=1]
        ret double %1
}

