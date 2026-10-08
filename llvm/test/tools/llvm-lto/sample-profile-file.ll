; REQUIRES: aarch64-registered-target
;
; RUN: llvm-as %S/Inputs/sample-profile-module.ll -o %t.o
; RUN: llvm-lto -exported-symbol=_foo -save-merged-module -o %t.lto.o \
; RUN:   -sample-profile-file=%S/Inputs/sample-profile.proftext %t.o
; RUN: llvm-dis %t.lto.o.merged.bc -o - | FileCheck %s

; CHECK-LABEL: define void @foo
; CHECK-SAME: !prof ![[ENTRY:[0-9]+]]
; CHECK: tail call void @bar(){{.*}}!prof ![[CALL:[0-9]+]]
; CHECK-DAG: ![[ENTRY]] = !{!"function_entry_count", i64 101}
; CHECK-DAG: ![[CALL]] = !{!"branch_weights", i32 101}
