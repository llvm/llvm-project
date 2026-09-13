; Test that a non-prevailing non-key COMDAT member does not make the whole
; COMDAT non-prevailing.
;
; input1 defines D0 in a self-keyed COMDAT. input2 defines D0 and D2 in a D5
; COMDAT, with D1 aliasing D2. When input1's D0 prevails, input2's D2 must
; remain a definition even though input2's D0 is non-prevailing.
;
; Also test the reverse input order, where the D5 D0 prevails and the
; self-keyed D0 COMDAT is discarded.
;
; RUN: llvm-as %p/Inputs/non-prevailing-comdat-member-a.ll -o %t.a.bc
; RUN: llvm-as %p/Inputs/non-prevailing-comdat-member-b.ll -o %t.b.bc
; RUN: llvm-lto2 run %t.a.bc %t.b.bc --save-temps -o %t.forward \
; RUN:   -r=%t.a.bc,_ZN1AIiED0Ev,px \
; RUN:   -r=%t.b.bc,_ZN1AIiED0Ev, \
; RUN:   -r=%t.b.bc,_ZN1AIiED1Ev,px \
; RUN:   -r=%t.b.bc,_ZN1AIiED2Ev,px
; RUN: llvm-dis %t.forward.0.0.preopt.bc -o - | FileCheck %s --check-prefix=FORWARD
; RUN: llvm-lto2 run %t.b.bc %t.a.bc --save-temps -o %t.reverse \
; RUN:   -r=%t.b.bc,_ZN1AIiED0Ev,px \
; RUN:   -r=%t.b.bc,_ZN1AIiED1Ev,px \
; RUN:   -r=%t.b.bc,_ZN1AIiED2Ev,px \
; RUN:   -r=%t.a.bc,_ZN1AIiED0Ev,
; RUN: llvm-dis %t.reverse.0.0.preopt.bc -o - | FileCheck %s --check-prefix=REVERSE
;
; FORWARD: $_ZN1AIiED5Ev = comdat any
; FORWARD-DAG: @_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev
; FORWARD-DAG: define weak_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat {
; FORWARD-DAG: define weak_odr void @_ZN1AIiED2Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev)
;
; REVERSE: $_ZN1AIiED5Ev = comdat any
; REVERSE-DAG: @_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev
; REVERSE-DAG: define weak_odr void @_ZN1AIiED2Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev)
; REVERSE-DAG: define weak_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev)
