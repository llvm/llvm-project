;; Test that the call type IDs in function summaries round trip through the
;; assembler and disassembler.
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

^0 = module: (path: "thinlto-call-type-ids-summary.o", hash: (0, 0, 0, 0, 0))
;; All fields.
^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 2, callTypeIdInfo: (callTypeId: 3430040695729964317, indirectCallTypeIds: (5970146889962867861, 11963068178503280946), hasCallWithoutTypeId: 1))))
;; Only the call type ID of the function.
^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, callTypeIdInfo: (callTypeId: 5970146889962867861))))
;; The fields can come in any order, and a missing call type ID of the function
;; is 0. A flag that is 0 is not printed.
^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1), insts: 3, calls: ((callee: ^2)), callTypeIdInfo: (hasCallWithoutTypeId: 0, indirectCallTypeIds: (5485679292981660066)))))
;; Call type IDs that are all empty are different from no call type IDs: the
;; function with GUID 5 has none.
^4 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1), insts: 1, callTypeIdInfo: (callTypeId: 0))))
^5 = gv: (guid: 5, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1), insts: 1)))

; CHECK: ^1 = gv: (guid: 1, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 2, callTypeIdInfo: (callTypeId: 3430040695729964317, indirectCallTypeIds: (5970146889962867861, 11963068178503280946), hasCallWithoutTypeId: 1))))
; CHECK: ^2 = gv: (guid: 2, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 1, callTypeIdInfo: (callTypeId: 5970146889962867861))))
; CHECK: ^3 = gv: (guid: 3, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 3, calls: ((callee: ^2)), callTypeIdInfo: (callTypeId: 0, indirectCallTypeIds: (5485679292981660066)))))
; CHECK: ^4 = gv: (guid: 4, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 1, callTypeIdInfo: (callTypeId: 0))))
; CHECK: ^5 = gv: (guid: 5, summaries: (function: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 1)))
