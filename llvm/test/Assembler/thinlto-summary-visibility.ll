;; Test visibility parsing in GlobalValueSummary::GVFlags.
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

^0 = module: (path: "thinlto-summary-visibility1.o", hash: (1369602428, 2747878711, 259090915, 2507395659, 1141468049))
^1 = module: (path: "thinlto-summary-visibility2.o", hash: (2998369023, 4283347029, 1195487472, 2757298015, 1852134156))

; CHECK:      ^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 10)))
; CHECK-NEXT: ^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external, visibility: protected, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 10)))
; CHECK-NEXT: ^4 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: external, visibility: hidden, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0), insts: 10)))

^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default), insts: 10)))
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: external, visibility: protected), insts: 10)))
^4 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: external, visibility: hidden, notEligibleToImport: 0), insts: 10)))
