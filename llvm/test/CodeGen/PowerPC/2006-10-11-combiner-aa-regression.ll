; RUN: llvm-as < %s | \
; RUN:   llc -march=ppc32 -combiner-alias-analysis | grep f5

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.2.0"
        %struct.Point = type { double, double, double }

define void @offset(%struct.Point* %pt, double %x, double %y, double %z) {
entry:
        %tmp = getelementptr %struct.Point* %pt, i32 0, i32 0           ; <double*> [#uses=2]
        %tmp.upgrd.1 = load double* %tmp                ; <double> [#uses=1]
        %tmp2 = add double %tmp.upgrd.1, %x             ; <double> [#uses=1]
        store double %tmp2, double* %tmp
        %tmp6 = getelementptr %struct.Point* %pt, i32 0, i32 1          ; <double*> [#uses=2]
        %tmp7 = load double* %tmp6              ; <double> [#uses=1]
        %tmp9 = add double %tmp7, %y            ; <double> [#uses=1]
        store double %tmp9, double* %tmp6
        %tmp13 = getelementptr %struct.Point* %pt, i32 0, i32 2         ; <double*> [#uses=2]
        %tmp14 = load double* %tmp13            ; <double> [#uses=1]
        %tmp16 = add double %tmp14, %z          ; <double> [#uses=1]
        store double %tmp16, double* %tmp13
        ret void
}

