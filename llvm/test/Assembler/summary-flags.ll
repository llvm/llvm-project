; ModuleID = 'tmp.bc'
source_filename = "tmp.bc"

; Test parsing of summary flas. Expect that flags value is the same after round-trip through
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s
; CHECK:       ^0 = module
; CHECK-NEXT:  ^1 = gv
; CHECK-NEXT:  ^2 = flags: 33

^0 = module: (path: "main.bc", hash: (3499594384, 1671013073, 3271036935, 1830411232, 59290952))
^1 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0), insts: 2)))
^2 = flags: 33
