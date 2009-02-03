; RUN: llvm-as < %s | llc -march=x86 -mcpu=i386 | \
; RUN:   grep {fucomi.*st.\[12\]}
; PR1012

define float @foo(float* %col.2.0) {
        %tmp = load float* %col.2.0             ; <float> [#uses=3]
        %tmp16 = fcmp olt float %tmp, 0.000000e+00              ; <i1> [#uses=1]
        %tmp20 = sub float -0.000000e+00, %tmp          ; <float> [#uses=1]
        %iftmp.2.0 = select i1 %tmp16, float %tmp20, float %tmp         ; <float> [#uses=1]
        ret float %iftmp.2.0
}

