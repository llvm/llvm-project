; RUN: llc -verify-machineinstrs < %s -march=ppc32 | not grep fmr

define double @test(float %F) {
        %F.upgrd.1 = fpext float %F to double           ; <double> [#uses=1]
        ret double %F.upgrd.1
}

