; RUN: rm -rf %t && split-file %s %t
; RUN: opt -S %t/main.ll -passes=lowertypetests -lowertypetests-summary-action=export \
; RUN:   -lowertypetests-read-summary=%t/summary.ll | FileCheck %s
;
;--- main.ll
; CHECK: @external_addrtaken = alias [8 x i8], ptr @[[JT:.*]]
; CHECK: @external_addrtaken_dso_local = dso_local alias [8 x i8], {{.*}}ptr @[[JT]]
; CHECK: @alias1 = alias [8 x i8], ptr @external_addrtaken
; CHECK: @alias2 = alias [8 x i8], ptr @external_addrtaken
; CHECK: @alias_dso_local = dso_local alias [8 x i8], ptr @external_addrtaken_dso_local
; CHECK-NOT: @alias3 = alias
; CHECK-NOT: @not_present

target triple = "x86_64-unknown-linux"

!cfi.functions = !{!0, !2, !3, !4, !7, !8}
!aliases = !{!5, !6, !9}

!0 = !{!"external_addrtaken", i8 0, i64 16594175687743574550, !1}
!1 = !{i64 0, !"typeid1"}
!2 = !{!"alias1", i8 0, i64 1062103744896965210, !1}
!3 = !{!"alias2", i8 0, i64 2510616090736846890, !1}
!4 = !{!"alias3", i8 0, i64 9766217518394409673, !1}
!5 = !{!"external_addrtaken", !"alias1", !"alias2"}
!6 = !{!"not_present", !"alias3"}
!7 = !{!"external_addrtaken_dso_local", i8 0, i64 4497425064003378793, !1}
!8 = !{!"alias_dso_local", i8 0, i64 15765058181946593837, !1}
!9 = !{!"external_addrtaken_dso_local", !"alias_dso_local"}

;--- summary.ll
^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 42,                   summaries: (function: (module: ^0, flags: (live: 1), insts: 1, refs: (^2, ^5), typeIdInfo: (typeTests: (14276520915468743435)))))
^2 = gv: (guid: 16594175687743574550, summaries: (function: (module: ^0, flags: (live: 1), insts: 1)))
^3 = gv: (guid: 1062103744896965210,  summaries: (alias:    (module: ^0, flags: (linkage: weak, live: 1), aliasee: ^2)))
^4 = gv: (guid: 2510616090736846890,  summaries: (alias:    (module: ^0, flags: (live: 1), aliasee: ^2)))
^5 = gv: (guid: 4497425064003378793,  summaries: (function: (module: ^0, flags: (live: 1, dsoLocal: 1), insts: 1)))
^6 = gv: (guid: 15765058181946593837, summaries: (alias:    (module: ^0, flags: (linkage: weak, live: 1, dsoLocal: 1), aliasee: ^5)))
