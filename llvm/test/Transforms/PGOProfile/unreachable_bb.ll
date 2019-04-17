; RUN: llvm-profdata merge %S/Inputs/unreachable_bb.proftext -o %t.profdata
; RUN: opt < %s -pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s --check-prefix=USE

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
  call void @bar()
  unreachable
return:
  ret void
}

declare void @bar()

;USE: !0 = !{i32 1, !"ProfileSummary", !1}
;USE: !1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
;USE: !2 = !{!"ProfileFormat", !"InstrProf"}
;USE: !3 = !{!"TotalCount", i64 0}


