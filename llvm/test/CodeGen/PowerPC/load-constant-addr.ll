; Should fold the ori into the lfs.
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | grep lfs
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep ori

define float @test() {
        %tmp.i = load float, ptr inttoptr (i32 186018016 to ptr)         ; <float> [#uses=1]
        ret float %tmp.i
}

