; Device globals are the motivating consumer: the host runtime resolves them
; through ELF symbols, so it needs the declared size rather than the padded one.
; Redzones grow with the object, so the two sizes can differ by a lot.

; RUN: opt < %s -passes=asan -S | FileCheck %s

target triple = "amdgpu7.00-amd-amdhsa"

@scalar = addrspace(1) global i32 7, align 4
@array = addrspace(1) global [64 x float] zeroinitializer, align 4
@huge = addrspace(1) global [1000000 x i8] zeroinitializer, align 1
@ro = addrspace(4) global i64 7, align 8

; CHECK: @scalar = addrspace(1) global { i32, [28 x i8] } {{.*}}!sanitize.unpadded.size ![[SCALAR:[0-9]+]]
; CHECK: @array = addrspace(1) global { [64 x float], [64 x i8] } {{.*}}!sanitize.unpadded.size ![[ARRAY:[0-9]+]]
; CHECK: @huge = addrspace(1) global { [1000000 x i8], [249984 x i8] } {{.*}}!sanitize.unpadded.size ![[HUGE:[0-9]+]]
; CHECK: @ro = addrspace(4) global { i64, [24 x i8] } {{.*}}!sanitize.unpadded.size ![[RO:[0-9]+]]

; CHECK-DAG: ![[SCALAR]] = !{i64 4}
; CHECK-DAG: ![[ARRAY]] = !{i64 256}
; CHECK-DAG: ![[HUGE]] = !{i64 1000000}
; CHECK-DAG: ![[RO]] = !{i64 8}
