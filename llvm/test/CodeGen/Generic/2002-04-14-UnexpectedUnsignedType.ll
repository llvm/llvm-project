; RUN: llc < %s

; This caused the backend to assert out with:
; SparcInstrInfo.cpp:103: failed assertion `0 && "Unexpected unsigned type"'
;

declare void @bar(ptr)

define void @foo() {
        %cast225 = inttoptr i64 123456 to ptr           ; <ptr> [#uses=1]
        call void @bar( ptr %cast225 )
        ret void
}
