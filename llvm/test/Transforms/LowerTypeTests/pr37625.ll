; RUN: opt -S %s -passes=lowertypetests -lowertypetests-summary-action=export \
; RUN:   -lowertypetests-read-summary=%s -lowertypetests-write-summary=%t/out.summary | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare !type !2 extern_weak void @external_addrtaken(i8)

!cfi.functions = !{!0, !1}

!0 = !{!"external_addrtaken", i8 2, i64 16594175687743574550, !2}
!1 = !{!"external_addrtaken", i8 0, i64 16594175687743574550, !2}
!2 = !{i64 0, !"typeid1"}

; CHECK-DAG: @external_addrtaken = alias [8 x i8], ptr @.cfi.jumptable

^0 = module: (path: "test.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 42,                   summaries: (function: (module: ^0, flags: (live: 1), insts: 1, refs: (^2), typeIdInfo: (typeTests: (14276520915468743435)))))
^2 = gv: (guid: 16594175687743574550, summaries: (function: (module: ^0, flags: (live: 1), insts: 1)))
