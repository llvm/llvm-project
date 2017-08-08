; Should fold the ori into the lfs.
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | grep lfs
; RUN: llc -verify-machineinstrs < %s -march=ppc32 | not grep ori

define float @test() {
        %tmp.i = load float, float* inttoptr (i32 186018016 to float*)         ; <float> [#uses=1]
        ret float %tmp.i
}

