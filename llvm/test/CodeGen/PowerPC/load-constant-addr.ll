; Should fold the ori into the lfs.
; RUN: llvm-as < %s | llc -march=ppc32 | grep lfs
; RUN: llvm-as < %s | llc -march=ppc32 | not grep ori

define float @test() {
        %tmp.i = load float* inttoptr (i32 186018016 to float*)         ; <float> [#uses=1]
        ret float %tmp.i
}

