; Threadgroup globals declared but not defined (PTX .shared variables, as
; agx-raise-to-air keeps them): Metal resolves no such symbol. A sized one is
; defined; a zero-sized one (CUDA's dynamic shared memory) becomes a
; threadgroup argument of each kernel using it, location index 0, whose
; length the host sets per dispatch.
; RUN: opt -mtriple=air -passes=air-legalize-unsupported-ir -S %s | FileCheck %s
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o /dev/null

; CHECK: @marker = internal addrspace(3) global i32 undef, align 4
; CHECK-NOT: @dyn_smem
@dyn_smem = external addrspace(3) global [0 x i8], align 16
@marker = external addrspace(3) global i32, align 4

; CHECK-LABEL: define void @reverse(ptr addrspace(1) %x, <3 x i32> %tidtg, ptr addrspace(3) %dyn_smem)
; CHECK: %[[A:.*]] = ptrtoint ptr addrspace(3) %dyn_smem to i32
; CHECK: add i32 %{{.*}}, %[[A]]
; CHECK: store i32 7, ptr addrspace(3) @marker
; CHECK: getelementptr i8, ptr addrspace(3) %dyn_smem, i32 4
define void @reverse(ptr addrspace(1) %x, <3 x i32> %tidtg) "air-kernel" {
  %t = extractelement <3 x i32> %tidtg, i64 0
  %o = shl i32 %t, 2
  %a = add i32 %o, ptrtoint (ptr addrspace(3) @dyn_smem to i32)
  %z = zext i32 %a to i64
  %p = inttoptr i64 %z to ptr addrspace(3)
  store i32 %t, ptr addrspace(3) %p, align 4
  store i32 7, ptr addrspace(3) @marker, align 4
  %q = getelementptr i8, ptr addrspace(3) @dyn_smem, i32 4
  %v = load i32, ptr addrspace(3) %q, align 4
  store i32 %v, ptr addrspace(1) %x, align 4
  ret void
}

; The pre-baked !air.kernel gains the argument (number 2), as a threadgroup
; buffer at threadgroup location index 0.
; CHECK: !air.kernel = !{![[K:[0-9]+]]}
; CHECK: ![[K]] = !{ptr @reverse, !{{[0-9]+}}, ![[ARGS:[0-9]+]]}
; CHECK: ![[ARGS]] = !{!{{[0-9]+}}, !{{[0-9]+}}, ![[TG:[0-9]+]]}
; CHECK: ![[TG]] = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dyn_smem"}
!air.kernel = !{!0}
!0 = !{ptr @reverse, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"x"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tidtg"}
