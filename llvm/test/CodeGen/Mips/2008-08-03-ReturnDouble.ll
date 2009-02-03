; Double return in abicall (default)
; RUN: llvm-as < %s | llc -march=mips
; PR2615

define double @main(...) {
entry:
        %retval = alloca double         ; <double*> [#uses=3]
        store double 0.000000e+00, double* %retval
        %r = alloca double              ; <double*> [#uses=1]
        load double* %r         ; <double>:0 [#uses=1]
        store double %0, double* %retval
        br label %return

return:         ; preds = %entry
        load double* %retval            ; <double>:1 [#uses=1]
        ret double %1
}

