; RUN: llc %s -O0 -march=sparc -mcpu=at697e -o - | FileCheck %s -check-prefix=FIXCALL
; RUN: llc %s -O0 -march=sparc -mcpu=leon2 -mattr=+fixcall -o - | FileCheck %s -check-prefix=FIXCALL

; RUN: llc %s -O0 -march=sparc -mcpu=at697e -mattr=-fixcall -o - | FileCheck %s -check-prefix=NO_FIXCALL
; RUN: llc %s -O0 -march=sparc -mcpu=leon2  -o - | FileCheck %s -check-prefix=NO_FIXCALL


; FIXCALL-LABEL:       	immediate_call_test
; FIXCALL:       	call 763288

; NO_FIXCALL-LABEL:     immediate_call_test
; NO_FIXCALL:       	call 2047583640
define void @immediate_call_test() nounwind {
entry:
        call void asm sideeffect "call $0", "i"(i32 2047583640) nounwind
        ret void
}



