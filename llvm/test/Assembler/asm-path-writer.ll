; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: ^0 = module: (path: ".\5Cf4folder\5Cabc.o", hash: (0, 0, 0, 0, 0))

^0 = module: (path: ".\5Cf4folder\5Cabc.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 15822663052811949562, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2)))
