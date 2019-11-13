; RUN: llvm-as %s -o %t.o
; RUN: llvm-cat -b -o %t2.o %t.o %t.o
; RUN: llvm-dis -o %t3 %t2.o
; RUN: llvm-as -o %t4.o %t3.0
; RUN: llvm-as -o %t5.o %t3.1
; RUN: cmp %t4.o %t5.o
; RUN: llvm-cat -b -o %t6.o %t5.o %t4.o
; RUN: llvm-dis -o %t7.o %t6.o
; RUN: diff %t7.o.0 %t7.o.1
; RUN: FileCheck < %t7.o.0 %s
; RUN: FileCheck < %t7.o.1 %s

; ModuleID = 'multi-summary-disassemble.o'

^0 = module: (path: "multi-summary-disassemble.ll", hash: (1369602428, 2747878711, 259090915, 2507395659, 1141468049))
^1 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: internal, notEligibleToImport: 0, live: 0, dsoLocal: 1), insts: 1)))
; CHECK: ^0 = module: (path:
; CHECK: ^1 = gv: (guid: 3, summaries: (function: (module: ^0,
