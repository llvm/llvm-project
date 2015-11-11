; RUN: llvm-as %S/only-needed-named-metadata.ll -o %t.bc
; RUN: llvm-as %S/Inputs/only-needed-named-metadata.ll -o %t2.bc

; Without -only-needed we should lazy link linkonce globals, and the
; metadata reference should not cause them to be linked.
; RUN: llvm-link -S %t2.bc %t.bc | FileCheck %s
; CHECK-NOT:@U_linkonce
; CHECK-NOT:@unused_linkonce()

; With -only-needed the metadata references should not cause any of the
; otherwise unreferenced globals to be linked. This also ensures that the
; metadata references don't provoke the module linker to create declarations,
; which are illegal for aliases and globals in comdats.
; Note that doing -only-needed with the comdat shown below leads to a only
; part of the comdat group being linked, which is not technically correct.
; RUN: llvm-link -S -only-needed %t2.bc %t.bc | FileCheck %s -check-prefix=ONLYNEEDED
; RUN: llvm-link -S -internalize -only-needed %t2.bc %t.bc | FileCheck %s -check-prefix=ONLYNEEDED
; ONLYNEEDED-NOT:@U
; ONLYNEEDED-NOT:@U_linkonce
; ONLYNEEDED-NOT:@unused()
; ONLYNEEDED-NOT:@unused_linkonce()
; ONLYNEEDED-NOT:@linkoncealias
; ONLYNEEDED-NOT:@linkoncefunc2()
; ONLYNEEDED-NOT:@weakalias
; ONLYNEEDED-NOT:@globalfunc1()
; ONLYNEEDED-NOT:@analias
; ONLYNEEDED-NOT:@globalfunc2()

@X = global i32 5
@U = global i32 6
@U_linkonce = linkonce_odr hidden global i32 6
define i32 @foo() { ret i32 7 }
define i32 @unused() { ret i32 8 }
define linkonce_odr hidden i32 @unused_linkonce() { ret i32 8 }
@linkoncealias = alias void (...), bitcast (void ()* @linkoncefunc2 to void (...)*)

@weakalias = weak alias void (...), bitcast (void ()* @globalfunc1 to void (...)*)
@analias = alias void (...), bitcast (void ()* @globalfunc2 to void (...)*)

define void @globalfunc1() #0 {
entry:
  ret void
}

define void @globalfunc2() #0 {
entry:
  ret void
}

$linkoncefunc2 = comdat any
define linkonce_odr void @linkoncefunc2() #0 comdat {
entry:
  ret void
}

!llvm.named = !{!0, !1, !2, !3, !4, !5, !6}
!0 = !{i32 ()* @unused}
!1 = !{i32* @U}
!2 = !{i32 ()* @unused_linkonce}
!3 = !{i32* @U_linkonce}
!4 = !{void (...)* @weakalias}
!5 = !{void (...)* @analias}
!6 = !{void (...)* @linkoncealias}
