; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 > %t
; RUN: grep fmsr %t | count 4
; RUN: grep fsitos %t
; RUN: grep fmrs %t | count 2
; RUN: grep fsitod %t
; RUN: grep fmrrd %t | count 3
; RUN: not grep fmdrr %t 
; RUN: grep fldd %t
; RUN: grep fuitod %t
; RUN: grep fuitos %t
; RUN: grep 1065353216 %t

define float @f(i32 %a) {
entry:
        %tmp = sitofp i32 %a to float           ; <float> [#uses=1]
        ret float %tmp
}

define double @g(i32 %a) {
entry:
        %tmp = sitofp i32 %a to double          ; <double> [#uses=1]
        ret double %tmp
}

define double @uint_to_double(i32 %a) {
entry:
        %tmp = uitofp i32 %a to double          ; <double> [#uses=1]
        ret double %tmp
}

define float @uint_to_float(i32 %a) {
entry:
        %tmp = uitofp i32 %a to float           ; <float> [#uses=1]
        ret float %tmp
}

define double @h(double* %v) {
entry:
        %tmp = load double* %v          ; <double> [#uses=1]
        ret double %tmp
}

define float @h2() {
entry:
        ret float 1.000000e+00
}

define double @f2(double %a) {
        ret double %a
}

define void @f3() {
entry:
        %tmp = call double @f5( )               ; <double> [#uses=1]
        call void @f4( double %tmp )
        ret void
}

declare void @f4(double)

declare double @f5()

