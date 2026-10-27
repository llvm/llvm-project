; !air.kernel describes each buffer with the type the kernel reads and writes
; through it: its MSL name, size and alignment. Only the parameters that
; air-system-values adds are system values; a kernel's own argument named
; like one (tidx) is not.
; RUN: llc -mtriple=air -air-portable %s -o - | FileCheck %s

; CHECK: !{i32 0, !"air.buffer", {{.*}}!"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"h"}
; CHECK: !{i32 1, !"air.buffer", {{.*}}!"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"i"}
; CHECK: !{i32 2, !"air.buffer", {{.*}}!"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"f2"}
; CHECK: !{i32 3, !"air.buffer", {{.*}}!"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"long", !"air.arg_name", !"l"}
; CHECK: !{i32 4, !"air.buffer", {{.*}}!"air.arg_type_name", !"int", !"air.arg_name", !"tidx"}
; CHECK-NOT: !"air.arg_name", !"tidx"}
; CHECK: !{i32 5, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
; CHECK-NOT: !"air.thread_position_in_grid"

declare { i32, i32, i32 } @air.thread_position_in_grid.v3i32()

define void @kernel(ptr addrspace(1) %h, ptr addrspace(1) %i, ptr addrspace(1) %f2, ptr addrspace(1) %l, i32 %tidx) "air-kernel" {
entry:
  %t = call { i32, i32, i32 } @air.thread_position_in_grid.v3i32()
  %tx = extractvalue { i32, i32, i32 } %t, 0
  %sum = add i32 %tx, %tidx
  store half 0xH3C00, ptr addrspace(1) %h
  store i32 %sum, ptr addrspace(1) %i
  store <2 x float> zeroinitializer, ptr addrspace(1) %f2
  store i64 1, ptr addrspace(1) %l
  ret void
}
