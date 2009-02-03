; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah

; The vload was getting memoized to the previous scalar load!

define void @VertexProgram2() {
        %xFloat0.688 = load float* null         ; <float> [#uses=0]
        %loadVector37.712 = load <4 x float>* null              ; <<4 x float>> [#uses=1]
        %inFloat3.713 = insertelement <4 x float> %loadVector37.712, float 0.000000e+00, i32 3          ; <<4 x float>> [#uses=1]
        store <4 x float> %inFloat3.713, <4 x float>* null
        unreachable
}

