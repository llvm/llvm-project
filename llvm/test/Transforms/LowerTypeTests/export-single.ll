; RUN: rm -rf %t && split-file %s %t
; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=export \
; RUN:   -lowertypetests-read-summary=%t/summary.ll -lowertypetests-write-summary=%t/out.summary %t/main.ll | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t/out.summary

;--- main.ll
@foo = constant i32 42, !type !0

!0 = !{i32 0, !"typeid1"}

; CHECK: [[G:@[0-9]+]] = private constant { i32 } { i32 42 }

; CHECK: @__typeid_typeid1_global_addr = hidden alias i8, ptr [[G]]
; CHECK: @foo = alias i32, ptr [[G]]

; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Single
; SUMMARY-NEXT:       SizeM1BitWidth:  0

;--- summary.ll
^0 = module: (path: "use.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 42, summaries: (function: (module: ^0, flags: (live: 1), insts: 1, typeIdInfo: (typeTests: (14276520915468743435)))))
