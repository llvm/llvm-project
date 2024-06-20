; Checks that llvm-as/llvm-dis correctly assembles/disassembles index with
; multiple summaries with single GUID.
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

source_filename = "index.bc"

^0 = module: (path: "[Regular LTO]", hash: (0, 0, 0, 0, 0))
; CHECK: ^0 = module: (path: "[Regular LTO]", hash: (0, 0, 0, 0, 0))
^1 = module: (path: "main.bc", hash: (3499594384, 1671013073, 3271036935, 1830411232, 59290952))
; CHECK-NEXT: ^1 = module: (path: "main.bc", hash: (3499594384, 1671013073, 3271036935, 1830411232, 59290952))
^2 = gv: (guid: 13351721993301222997, summaries: (function: (module: ^1, flags: (linkage: linkonce_odr, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1), function: (module: ^1, flags: (linkage: available_externally, notEligibleToImport: 1, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1)))
; CHECK-NEXT: ^2 = gv: (guid: 13351721993301222997, summaries: (function: (module: ^1, flags: (linkage: linkonce_odr, visibility: default, notEligibleToImport: 0, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1), function: (module: ^1, flags: (linkage: available_externally, visibility: default, notEligibleToImport: 1, live: 1, dsoLocal: 1, canAutoHide: 0, importType: definition), insts: 1)))
