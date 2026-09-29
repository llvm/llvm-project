; Test that we annotate entire program's summary and not just this module's
; RUN: llvm-profdata merge %S/Inputs/PR28219.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S | FileCheck %s

define i32 @bar() {
entry:
  ret i32 1
}
; CHECK-DAG: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
; CHECK-DAG: {{![0-9]+}} = !{!"NumFunctions", i64 2}
; CHECK-DAG: {{![0-9]+}} = !{!"MaxFunctionCount", i64 3}

