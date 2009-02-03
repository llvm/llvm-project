;All this should do is not crash
;RUN: llvm-as < %s | llc -march=alpha

target datalayout = "e-p:64:64"
target triple = "alphaev67-unknown-linux-gnu"

define void @_ZNSt13basic_filebufIcSt11char_traitsIcEE22_M_convert_to_externalEPcl(i32 %f) {
entry:
        %tmp49 = alloca i8, i32 %f              ; <i8*> [#uses=0]
        %tmp = call i32 null( i8* null, i8* null, i8* null, i8* null, i8* null, i8* null, i8* null )               ; <i32> [#uses=0]
        ret void
}

