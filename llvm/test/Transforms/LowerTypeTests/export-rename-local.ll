; RUN: rm -rf %t && split-file %s %t
; RUN: opt -S %t/main.ll -passes=lowertypetests -lowertypetests-summary-action=export \
; RUN:   -lowertypetests-read-summary=%t/summary.ll | FileCheck %s

;--- main.ll
; CHECK: define internal void @external_addrtaken.1()
; CHECK: declare {{.*}} void @external_addrtaken.cfi()

target triple = "x86_64-unknown-linux"

define internal void @external_addrtaken() !type !1 {
  ret void
}

!cfi.functions = !{!0}

!0 = !{!"external_addrtaken", i8 0, i64 16594175687743574550, !1}
!1 = !{i64 0, !"typeid1"}

;--- summary.ll
^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 42,                   summaries: (function: (module: ^0, flags: (live: 1), insts: 1, refs: (^2), typeIdInfo: (typeTests: (14276520915468743435)))))
^2 = gv: (guid: 16594175687743574550, summaries: (function: (module: ^0, flags: (live: 1), insts: 1)))
