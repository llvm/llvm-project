; RUN: rm -rf %t && split-file %s %t
; RUN: opt -S %t/main.ll -passes=lowertypetests -lowertypetests-summary-action=export \
; RUN:   -lowertypetests-read-summary=%t/summary.ll | FileCheck %s
;
;--- main.ll
; CHECK: module asm
; CHECK-NEXT: ".symver external_addrtaken, alias1"
; CHECK-NOT: .symver external_addrtaken2
; CHECK-NOT: .symver not_exported

target triple = "x86_64-unknown-linux"

!cfi.functions = !{!0, !1}
!symvers = !{!3, !4}

!0 = !{!"external_addrtaken", i8 0, i64 16594175687743574550, !2}
!1 = !{!"external_addrtaken2", i8 0, i64 2415377257478301385, !2}
!2 = !{i64 0, !"typeid1"}
!3 = !{!"external_addrtaken", !"alias1"}
!4 = !{!"not_exported", !"alias2"}

;--- summary.ll
^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 42,                   summaries: (function: (module: ^0, flags: (live: 1), insts: 1, refs: (^2, ^3), typeIdInfo: (typeTests: (14276520915468743435)))))
^2 = gv: (guid: 16594175687743574550, summaries: (function: (module: ^0, flags: (live: 1), insts: 1)))
^3 = gv: (guid: 2415377257478301385,  summaries: (function: (module: ^0, flags: (live: 1), insts: 1)))
^4 = gv: (guid: 1062103744896965210,  summaries: (alias:    (module: ^0, flags: (linkage: weak, live: 1), aliasee: ^2)))
