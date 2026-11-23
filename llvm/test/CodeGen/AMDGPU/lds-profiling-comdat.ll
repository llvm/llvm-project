; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa < %s | FileCheck %s
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -filetype=obj < %s | llvm-readelf -gsW - | FileCheck -check-prefix=ELF %s

; The profiling clone of a kernel in a comdat and the counters of the clone are
; in a group of their own, so that a linker keeps or discards them together.

@lds = internal addrspace(3) global [64 x i32] poison, align 4

$k = comdat any

; CHECK: .section .text.k,"axG",@progbits,k,comdat
; CHECK: {{^}}k:
; CHECK: .section .text.k.prof,"axG",@progbits,k.prof,comdat
; CHECK: .weak k.prof
; CHECK: {{^}}k.prof:
; CHECK: s_add_nc_u64 {{.*}}, k.prof.counters@rel64+4
; CHECK: .section .bss.k.prof.counters,"awG",@nobits,k.prof,comdat
; CHECK-NEXT: .p2align 2
; CHECK-NEXT: .weak k.prof.counters
; CHECK-NEXT: .protected k.prof.counters
; CHECK-NEXT: .type k.prof.counters,@object
; CHECK-NEXT: k.prof.counters:
; CHECK-NEXT: .zero 20
; CHECK-NEXT: .size k.prof.counters, 20

; ELF: {{[0-9a-f]+}} 20 OBJECT WEAK PROTECTED {{[0-9]+}} k.prof.counters
; ELF: COMDAT group section [{{.*}}] `.group' [k] contains
; ELF: .text.k
; ELF: COMDAT group section [{{.*}}] `.group' [k.prof] contains
; ELF-DAG: .text.k.prof
; ELF-DAG: .bss.k.prof.counters
define linkonce_odr protected amdgpu_kernel void @k(i32 %i, i32 %v) comdat {
  %p = getelementptr inbounds [64 x i32], ptr addrspace(3) @lds, i32 0, i32 %i
  store i32 %v, ptr addrspace(3) %p, align 4
  ret void
}
