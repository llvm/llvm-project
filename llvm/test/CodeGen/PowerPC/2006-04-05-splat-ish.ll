; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin8 -mcpu=g5 | \
; RUN:   grep "vspltish v.*, 10"

define void @test(<8 x i16>* %P) {
        %tmp = load <8 x i16>, <8 x i16>* %P               ; <<8 x i16>> [#uses=1]
        %tmp1 = add <8 x i16> %tmp, < i16 10, i16 10, i16 10, i16 10, i16 10, i16 10, i16 10, i16 10 >          ; <<8 x i16>> [#uses=1]
        store <8 x i16> %tmp1, <8 x i16>* %P
        ret void
}

