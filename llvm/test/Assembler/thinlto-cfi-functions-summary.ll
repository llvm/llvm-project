;; Test that the CFI function entries of a combined summary (the names of the
;; functions with CFI jump table entries, with the GUIDs that ThinLTO uses for
;; them) round trip through the assembler and disassembler.
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s
;; Check that the entries are also accepted (and skipped) when the summary is
;; not parsed into an index.
; RUN: opt %s -disable-output

; CHECK: ^0 = module: (path: "main.o", hash: (0, 0, 0, 0, 0))
; CHECK: ^1 = gv: (guid: 13146401226427987378, summaries: (function: (module: ^0, {{.*}}
; CHECK: ^2 = gv: (guid: 14740650423002898831, summaries: (function: (module: ^0, {{.*}}
; CHECK: ^3 = gv: (guid: 16617365750458309383, summaries: (function: (module: ^0, {{.*}}
; CHECK: ^4 = cfiFunctionDefs: ((guid: 14740650423002898831, name: "f"), (guid: 16617365750458309383, name: "f_dso_local"))
; CHECK: ^5 = cfiFunctionDecls: ((guid: 13146401226427987378, name: "g"), (guid: 13146401226427987378, name: "g.llvm.123"))
; CHECK: ^6 = blockcount: 0

^0 = module: (path: "main.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 13146401226427987378, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1)))
^2 = gv: (guid: 14740650423002898831, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1)))
^3 = gv: (guid: 16617365750458309383, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1)))
^4 = cfiFunctionDefs: ((guid: 14740650423002898831, name: "f"), (guid: 16617365750458309383, name: "f_dso_local"))
;; Multiple names may share a ThinLTO GUID.
^5 = cfiFunctionDecls: ((guid: 13146401226427987378, name: "g"), (guid: 13146401226427987378, name: "g.llvm.123"))
^6 = blockcount: 0
