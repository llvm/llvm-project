; RUN: opt < %s -passes='function(instcombine),globalopt' -S | FileCheck %s
; CHECK: internal fastcc float @foo

define internal float @foo() {
        ret float 0.000000e+00
}

define float @bar() {
        %tmp1 = call float (...) @foo( )
        %tmp2 = fmul float %tmp1, 1.000000e+01           ; <float> [#uses=1]
        ret float %tmp2
}
