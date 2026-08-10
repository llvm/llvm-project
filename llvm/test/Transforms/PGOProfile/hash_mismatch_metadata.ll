; RUN: llvm-profdata merge %S/Inputs/hash_mismatch_metadata.proftext -o %t.profdata
; RUN: opt < %s -mtriple=x86_64-linux-gnu -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s

define void @foo() !annotation !1 {
entry:
   ret void
}

define void @bar() !annotation !2 {
entry:
   ret void
}

!1 = !{!"fake_metadata"}
!2 = !{!"instr_prof_hash_mismatch"}

; CHECK-DAG: !{{[0-9]+}} = !{!"fake_metadata", !"instr_prof_hash_mismatch"}
; CHECK-DAG: !{{[0-9]+}} = !{!"instr_prof_hash_mismatch"}
