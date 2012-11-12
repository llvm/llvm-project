; RUN: llc < %s -march=x86  | not grep set

declare i1 @llvm.isunordered.f32(float, float)

define float @cmp(float %A, float %B, float %C, float %D) nounwind {
entry:
        %tmp.1 = fcmp uno float %A, %B          ; <i1> [#uses=1]
        %tmp.2 = fcmp oge float %A, %B          ; <i1> [#uses=1]
        %tmp.3 = or i1 %tmp.1, %tmp.2           ; <i1> [#uses=1]
        %tmp.4 = select i1 %tmp.3, float %C, float %D           ; <float> [#uses=1]
        ret float %tmp.4
}

