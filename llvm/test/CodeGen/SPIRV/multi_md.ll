;; Check duplicate operands in opencl.ocl.version metadata is accepted without
;; assertion.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

%struct.my_struct_t = type { i8, i32 }

@var = addrspace(1) global %struct.my_struct_t { i8 97, i32 42 }, align 4

define spir_kernel void @__OpenCL_writer_kernel(i8 zeroext %c, i32 %i) {
entry:
  %c.addr = alloca i8, align 1
  %i.addr = alloca i32, align 4
  store i8 %c, ptr %c.addr, align 1
  store i32 %i, ptr %i.addr, align 4
  %0 = load i8, ptr %c.addr, align 1
  store i8 %0, ptr addrspace(1) @var, align 1
  %1 = load i32, ptr %i.addr, align 4
  store i32 %1, ptr addrspace(1) getelementptr inbounds (%struct.my_struct_t, ptr addrspace(1) @var, i32 0, i32 1), align 4
  ret void
}

define spir_kernel void @__OpenCL_reader_kernel(ptr addrspace(1) %C, ptr addrspace(1) %I) {
entry:
  %C.addr = alloca ptr addrspace(1), align 8
  %I.addr = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %C, ptr %C.addr, align 8
  store ptr addrspace(1) %I, ptr %I.addr, align 8
  %0 = load i8, ptr addrspace(1) @var, align 1
  %1 = load ptr addrspace(1), ptr %C.addr, align 8
  store i8 %0, ptr addrspace(1) %1, align 1
  %2 = load i32, ptr addrspace(1) getelementptr inbounds (%struct.my_struct_t, ptr addrspace(1) @var, i32 0, i32 1), align 4
  %3 = load ptr addrspace(1), ptr %I.addr, align 8
  store i32 %2, ptr addrspace(1) %3, align 4
  ret void
}

;; "cl_images" should be encoded as BasicImage capability,
;; but images are not used in this test case, so this capability is not required.
; CHECK-NOT: OpExtension "cl_images"
; CHECK-DAG: OpSourceExtension "cl_khr_int64_base_atomics"
; CHECK-DAG: OpSourceExtension "cl_khr_int64_extended_atomics"
; CHECK:     OpSource OpenCL_C 200000

!opencl.ocl.version = !{!13, !13}
!opencl.used.extensions = !{!24, !25}

!13 = !{i32 2, i32 0}
!24 = !{!"cl_khr_int64_base_atomics"}
!25 = !{!"cl_khr_int64_base_atomics", !"cl_khr_int64_extended_atomics"}
