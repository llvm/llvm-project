; RUN: llvm-split -enable-split-module-CG=true -j10 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; should only produce 2 output files (N capped to EntryFuncs.size()=2)

; CHECK0: define void @foo()
; CHECK1: define void @bar()

define void @foo() { ret void }
define void @bar() { ret void }
