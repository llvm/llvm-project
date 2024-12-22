; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; This checks for a bug in the DAG combiner where a SETCC was created with
; an illegal return type. Make sure it compiles successfully.
; CHECK: r0 = cmp.eq(r0,##-2147483648)

define i32 @f0(i32 %a0) #0 {
entry:
   %v0 = sdiv i32 %a0, -2147483648
   ret i32 %v0
}

attributes #0 = { noinline nounwind "target-cpu"="hexagonv60" }
