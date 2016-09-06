; Check the linkage types in both the per-module and combined summaries.
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-lto -thinlto -o %t2 %t.o
; RUN: llvm-bcanalyzer -dump %t2.thinlto.bc | FileCheck %s --check-prefix=COMBINED

; CHECK: <PERMODULE {{.*}} op1=16
; COMBINED-DAG: <COMBINED {{.*}} op2=16
define void @functionWithSection() section "some_section" {
    ret void
}
