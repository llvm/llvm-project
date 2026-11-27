; Test that a non-prevailing non-key COMDAT member does not make the whole
; COMDAT non-prevailing.
;
; a.ll defines D0 in a self-keyed COMDAT. b.ll defines D0 and D2 in a D5
; COMDAT, with D1 aliasing D2. When a.ll's D0 prevails, b.ll's D2 must
; remain a definition even though b.ll's D0 is non-prevailing.
;
; Also test the reverse input order, where the D5 D0 prevails and the
; self-keyed D0 COMDAT is discarded.
;
; RUN: split-file %s %t
; RUN: llvm-as %t/a.ll -o %t/a.bc
; RUN: llvm-as %t/b.ll -o %t/b.bc
; RUN: llvm-lto2 run %t/a.bc %t/b.bc --save-temps -o %t/forward \
; RUN:   -r=%t/a.bc,_ZN1AIiED0Ev,px \
; RUN:   -r=%t/b.bc,_ZN1AIiED0Ev, \
; RUN:   -r=%t/b.bc,_ZN1AIiED1Ev,px \
; RUN:   -r=%t/b.bc,_ZN1AIiED2Ev,px
; RUN: llvm-dis %t/forward.0.0.preopt.bc -o - | FileCheck %s --check-prefix=FORWARD
; RUN: llvm-lto2 run %t/b.bc %t/a.bc --save-temps -o %t/reverse \
; RUN:   -r=%t/b.bc,_ZN1AIiED0Ev,px \
; RUN:   -r=%t/b.bc,_ZN1AIiED1Ev,px \
; RUN:   -r=%t/b.bc,_ZN1AIiED2Ev,px \
; RUN:   -r=%t/a.bc,_ZN1AIiED0Ev,
; RUN: llvm-dis %t/reverse.0.0.preopt.bc -o - | FileCheck %s --check-prefix=REVERSE
;
; In mixed LTO, a.ll is a ThinLTO module, so the regular LTO combined module
; has no prevailing D0 definition. b.ll's available_externally D0 is linked
; into it with its COMDAT cleared, while D1 and D2 must remain definitions.
; RUN: opt -module-summary %t/a.ll -o %t/a.thin.bc
; RUN: llvm-lto2 run %t/a.thin.bc %t/b.bc --save-temps -o %t/mixed-forward \
; RUN:   -r=%t/a.thin.bc,_ZN1AIiED0Ev,px \
; RUN:   -r=%t/b.bc,_ZN1AIiED0Ev, \
; RUN:   -r=%t/b.bc,_ZN1AIiED1Ev,px \
; RUN:   -r=%t/b.bc,_ZN1AIiED2Ev,px
; RUN: llvm-dis %t/mixed-forward.0.0.preopt.bc -o - | FileCheck %s --check-prefix=MIXED
; RUN: llvm-lto2 run %t/b.bc %t/a.thin.bc --save-temps -o %t/mixed-reverse \
; RUN:   -r=%t/b.bc,_ZN1AIiED0Ev,px \
; RUN:   -r=%t/b.bc,_ZN1AIiED1Ev,px \
; RUN:   -r=%t/b.bc,_ZN1AIiED2Ev,px \
; RUN:   -r=%t/a.thin.bc,_ZN1AIiED0Ev,
; RUN: llvm-dis %t/mixed-reverse.0.0.preopt.bc -o - | FileCheck %s --check-prefix=REVERSE
;
; FORWARD-DAG: $_ZN1AIiED0Ev = comdat any
; FORWARD-DAG: $_ZN1AIiED5Ev = comdat any
; FORWARD-DAG: @_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev
; FORWARD-DAG: define weak_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat {
; FORWARD-DAG: define weak_odr void @_ZN1AIiED2Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev)
;
; REVERSE-NOT: $_ZN1AIiED0Ev
; REVERSE: $_ZN1AIiED5Ev = comdat any
; REVERSE-NOT: $_ZN1AIiED0Ev
; REVERSE-DAG: @_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev
; REVERSE-DAG: define weak_odr void @_ZN1AIiED2Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev)
; REVERSE-DAG: define weak_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev)
; REVERSE-NOT: $_ZN1AIiED0Ev
;
; MIXED-NOT: $_ZN1AIiED0Ev
; MIXED: $_ZN1AIiED5Ev = comdat any
; MIXED-NOT: $_ZN1AIiED0Ev
; MIXED-DAG: @_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev
; MIXED-DAG: define available_externally void @_ZN1AIiED0Ev(ptr %this) unnamed_addr {
; MIXED-DAG: define weak_odr void @_ZN1AIiED2Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev)
; MIXED-NOT: $_ZN1AIiED0Ev

;--- a.ll
target datalayout = "e-m:e-p270:32:32:32-p271:32:32:32-p272:64:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED0Ev = comdat any

define linkonce_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat {
  ret void
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32:32-p271:32:32:32-p272:64:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZN1AIiED5Ev = comdat any

@_ZN1AIiED1Ev = weak_odr unnamed_addr alias void (ptr), ptr @_ZN1AIiED2Ev

define weak_odr void @_ZN1AIiED2Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
  ret void
}

define weak_odr void @_ZN1AIiED0Ev(ptr %this) unnamed_addr comdat($_ZN1AIiED5Ev) {
  ret void
}
