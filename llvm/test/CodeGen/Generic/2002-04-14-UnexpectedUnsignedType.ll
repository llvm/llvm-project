; RUN: llc < %s

; This caused the backend to assert out with:
; SparcInstrInfo.cpp:103: failed assertion `0 && "Unexpected unsigned type"'
;

declare void @bar(i8*)

define void @foo() {
        %cast225 = inttoptr i64 123456 to i8*           ; <i8*> [#uses=1]
        call void @bar( i8* %cast225 )
        ret void
}
