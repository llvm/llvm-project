;; Test that we get an error for an unknown field in the call type IDs of a
;; function summary.
; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

^0 = module: (path: "thinlto-bad-call-type-ids-summary.o", hash: (0, 0, 0, 0, 0))
; CHECK: [[@LINE+1]]:{{[0-9]+}}: error: invalid callTypeIdInfo field
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, callTypeIdInfo: (typeTests: (1)))))
