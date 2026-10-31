; The only use of "typeid1" is in a dead function. Export nothing.

; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=export \
; RUN:   -lowertypetests-read-summary=%s -lowertypetests-write-summary=%t/out.summary %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t/out.summary
@foo = constant i32 42, !type !0

!0 = !{i32 0, !"typeid1"}

; CHECK-NOT: @__typeid_typeid1_global_addr =

; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT: WithGlobalValueDeadStripping: true
; SUMMARY-NEXT: ...

^0 = module: (path: "use.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 42, summaries: (function: (module: ^0, flags: (live: 0), insts: 1, typeIdInfo: (typeTests: (14276520915468743435)))))
^2 = flags: 1
