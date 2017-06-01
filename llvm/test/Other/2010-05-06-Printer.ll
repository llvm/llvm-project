; RUN: llc -O2 -print-after-all < %s 2>/dev/null
; RUN: llc -O2 -print-after-all < %s 2>&1 | FileCheck %s --check-prefix=ALL
; RUN: llc -O2 -print-after-all -filter-print-funcs=foo < %s 2>&1 | FileCheck %s --check-prefix=FOO
; REQUIRES: default_triple
define void @tester(){
  ret void
}

define void @foo(){
  ret void
}

;ALL: define void @tester()
;ALL: define void @foo()
;ALL: ModuleID =

;FOO: IR Dump After
;FOO-NEXT: define void @foo()
;FOO-NOT: define void @tester
