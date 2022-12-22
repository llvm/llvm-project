; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon-unknown--elf"

@g0 = external thread_local(initialexec) global i32
@g1 = external thread_local(initialexec) global i32

; CHECK-DAG: r{{[0-9]+}} = memw(##g0@IE)
; CHECK-DAG: r{{[0-9]+}} = memw(##g1@IE)
define i32 @f0() {
b0:
  %v0 = load i32, ptr @g1, align 4
  store i32 %v0, ptr @g0, align 4
  ret i32 0
}
