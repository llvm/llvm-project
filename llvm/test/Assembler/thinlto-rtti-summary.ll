; RUN: llvm-as %s -o - | llvm-dis -o %t.ll
; RUN: grep "^\^" %s >%t2
; RUN: grep "^\^" %t.ll >%t3
; Expect that the summary information is the same after round-trip through
; llvm-as and llvm-dis.
; RUN: diff -b %t2 %t3

target triple = "aarch64-unknown-linux-gnu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@_ZTSxxx = external global ptr
@_ZTIxxx = external global ptr
@xxx = constant [1 x ptr] [ptr @_ZTIxxx]

^0 = module: (path: "<stdin>", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "_ZTIxxx") ; guid = 2928584540419986814
^2 = gv: (name: "xxx", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0, importType: definition), varFlags: (readonly: 1, writeonly: 0, constant: 1), refs: (^1)))) ; guid = 5616283335571169781
^3 = gv: (name: "_ZTSxxx") ; guid = 16805677846636166078
^4 = typeidMayBeAccessed: (name: "_ZTSxxx")
^5 = blockcount: 0
