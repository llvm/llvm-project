; Test sectionInfo parsing
; RUN: llvm-as %s -o - | llvm-dis -o - | FileCheck %s

@a = internal global i32 0, section ".foodata", align 4, !guid !0
@b = dso_local global ptr @a, section ".foodata", align 8, !guid !1

define dso_local void @g() section ".footext" !guid !8 {
entry:
  ret void
}

!0 = !{i64 -4514776715853495485}
!1 = !{i64 -1427730249719747694}
!8 = !{i64 -5300342847281564238}

^0 = module: (path: "[Regular LTO]", hash: (0, 0, 0, 0, 0))
^1 = gv: (name: "g", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), insts: 1, funcFlags: (readNone: 0, readOnly: 0, noRecurse: 0, returnDoesNotAlias: 0, noInline: 1, alwaysInline: 0, noUnwind: 1, mayThrow: 0, hasUnknownCall: 0, mustBeUnreachable: 0))), sectionInfo: (sectionName: ".footext", outputSectionName: "", keep: false)) ; guid = 13146401226427987378
^2 = gv: (name: "a", summaries: (variable: (module: ^0, flags: (linkage: internal, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), varFlags: (readonly: 1, writeonly: 1, constant: 0))), sectionInfo: (sectionName: ".foodata", outputSectionName: "", keep: false)) ; guid = 13931967357856056131
^3 = gv: (name: "b", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 1, live: 0, dsoLocal: 1, canAutoHide: 0, importType: definition, noRenameOnPromotion: 0), varFlags: (readonly: 1, writeonly: 1, constant: 0), refs: (^2))), sectionInfo: (sectionName: ".foodata", outputSectionName: "", keep: false)) ; guid = 17019013823989803922
^4 = blockcount: 0

; CHECK: sectionInfo: (sectionName: ".footext", outputSectionName: "", keep: false)
; CHECK: sectionInfo: (sectionName: ".foodata", outputSectionName: "", keep: false)
; CHECK: sectionInfo: (sectionName: ".foodata", outputSectionName: "", keep: false)
