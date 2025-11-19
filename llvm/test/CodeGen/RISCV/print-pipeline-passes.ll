; RUN: opt -mtriple=riscv32 -S -passes='default<O0>' -print-pipeline-passes < %s | FileCheck --check-prefix=O0 %s
; RUN: opt -mtriple=riscv64 -S -passes='default<O0>' -print-pipeline-passes < %s | FileCheck --check-prefix=O0 %s
; RUN: opt -mtriple=riscv32 -S -passes='default<O2>' -print-pipeline-passes < %s | FileCheck %s
; RUN: opt -mtriple=riscv64 -S -passes='default<O2>' -print-pipeline-passes < %s | FileCheck %s

; CHECK: loop-idiom-vectorize
; O0-NOT: loop-idiom-vectorize

define void @foo() {
entry:
  ret void
}
