; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s

; CHECK: <GLOBALVAL_SUMMARY_BLOCK
; ensure @f is marked readnone
; CHECK:  <PERMODULE {{.*}} op0=0 {{.*}} op3=1
; ensure @g is marked readonly
; CHECK:  <PERMODULE {{.*}} op0=1 {{.*}} op3=2
; ensure @h is marked norecurse
; CHECK:  <PERMODULE {{.*}} op0=2 {{.*}} op3=4
; ensure @i is marked returndoesnotalias
; CHECK:  <PERMODULE {{.*}} op0=3 {{.*}} op3=8

define void @f() readnone {
   ret void
}
define void @g() readonly {
   ret void
}
define void @h() norecurse {
   ret void
}

define noalias i8* @i() {
   %r = alloca i8
   ret i8* %r
}
