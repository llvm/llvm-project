; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s

define void @test(<8 x i16>* %P) {
; CHECK: vspltish {{[0-9]+}}, 10
        %tmp = load <8 x i16>, <8 x i16>* %P               ; <<8 x i16>> [#uses=1]
        %tmp1 = add <8 x i16> %tmp, < i16 10, i16 10, i16 10, i16 10, i16 10, i16 10, i16 10, i16 10 >          ; <<8 x i16>> [#uses=1]
        store <8 x i16> %tmp1, <8 x i16>* %P
        ret void
}

