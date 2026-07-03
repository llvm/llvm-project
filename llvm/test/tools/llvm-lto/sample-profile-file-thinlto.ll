; REQUIRES: aarch64-registered-target
;
; RUN: opt -module-summary %S/Inputs/sample-profile-module.ll -o %t.o
; RUN: llvm-lto -thinlto-action=optimize -o %t.opt.bc \
; RUN:   -sample-profile-file=%S/Inputs/sample-profile.proftext %t.o
; RUN: llvm-dis %t.opt.bc -o - | FileCheck %s

; CHECK-LABEL: define void @foo
; CHECK-SAME: !prof ![[ENTRY:[0-9]+]]
; CHECK: tail call void @bar(){{.*}}!prof ![[CALL:[0-9]+]]
; CHECK-DAG: ![[ENTRY]] = !{!"function_entry_count", i64 101}
; CHECK-DAG: ![[CALL]] = !{!"branch_weights", i32 101}
