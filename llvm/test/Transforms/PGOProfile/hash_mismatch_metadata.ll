; RUN: llvm-profdata merge %S/Inputs/hash_mismatch_metadata.proftext -o %t.profdata
; RUN: opt < %s -mtriple=x86_64-linux-gnu -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -S | FileCheck %s --check-prefixes=CHECK,COLD-DEFAULT
; RUN: opt < %s -mtriple=x86_64-linux-gnu -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -pgo-use-treat-zero-mismatch-as-cold=false -S | FileCheck %s --check-prefixes=CHECK,NO-COLD

; Non-zero hash mismatch: must NOT be marked cold or given entry_count 0.
; CHECK-NOT: Function Attrs:{{.*}}cold
; CHECK-LABEL: define void @foo()
; CHECK-NOT: !prof
; CHECK-SAME: !annotation ![[FOO_ANN:[0-9]+]] {
define void @foo() !annotation !1 {
entry:
   ret void
}

; CHECK-LABEL: define void @bar()
; CHECK-NOT: !prof
; CHECK-SAME: !annotation ![[BAR_ANN:[0-9]+]] {
define void @bar() !annotation !2 {
entry:
   ret void
}

; All-zero hash mismatch: marked cold + minsize + entry_count 0 by default,
; while still preserving instr_prof_hash_mismatch annotation.
; COLD-DEFAULT: Function Attrs: cold minsize
; COLD-DEFAULT-NEXT: define void @zero_mismatch() #{{[0-9]+}} !prof ![[ZERO_ENTRY:[0-9]+]] !annotation ![[BAR_ANN]] {
; NO-COLD-LABEL: define void @zero_mismatch()
; NO-COLD-NOT: !prof
; NO-COLD-SAME: !annotation ![[BAR_ANN]] {
define void @zero_mismatch() {
entry:
   ret void
}

; Explicit hot attribute takes precedence over 0-count mismatch.
; CHECK: Function Attrs: hot
; CHECK-NEXT: define void @zero_mismatch_hot_attr() #{{[0-9]+}} !annotation ![[BAR_ANN]] {
define void @zero_mismatch_hot_attr() hot {
entry:
   ret void
}

!1 = !{!"fake_metadata"}
!2 = !{!"instr_prof_hash_mismatch"}

; COLD-DEFAULT-DAG: ![[ZERO_ENTRY]] = !{!"function_entry_count", i64 0}
; CHECK-DAG: ![[FOO_ANN]] = !{!"fake_metadata", !"instr_prof_hash_mismatch"}
; CHECK-DAG: ![[BAR_ANN]] = !{!"instr_prof_hash_mismatch"}
