; Test checks that kernels are being split by attached module-id metadata and
; used functions are being moved with kernels that use them.

; RUN: llvm-split -split-by-category=module-id -S < %s -o %t
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-TU0,CHECK
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-TU1,CHECK

; CHECK-TU1-NOT: @GV
; CHECK-TU0: @GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4
@GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4

; CHECK-TU0-NOT: define dso_local spir_kernel void @TU1_kernelA
; CHECK-TU1: define dso_local spir_kernel void @TU1_kernelA
define dso_local spir_kernel void @TU1_kernelA() #0 {
entry:
; CHECK-TU1: call spir_func void @func1_TU1()
  call spir_func void @func1_TU1()
  ret void
}

; CHECK-TU0-NOT: define {{.*}} spir_func void @func1_TU1()
; CHECK-TU1: define {{.*}} spir_func void @func1_TU1()
define dso_local spir_func void @func1_TU1() {
entry:
; CHECK-TU1: call spir_func void @func2_TU1()
  call spir_func void @func2_TU1()
  ret void
}

; CHECK-TU0-NOT: define {{.*}} spir_func void @func2_TU1()
; CHECK-TU1: define {{.*}} spir_func void @func2_TU1()
define linkonce_odr dso_local spir_func void @func2_TU1() {
entry:
  ret void
}

; CHECK-TU0-NOT: define dso_local spir_kernel void @TU1_kernelB()
; CHECK-TU1: define dso_local spir_kernel void @TU1_kernelB()
define dso_local spir_kernel void @TU1_kernelB() #0 {
entry:
; CHECK-TU1: call spir_func void @func3_TU1()
  call spir_func void @func3_TU1()
  ret void
}

; CHECK-TU0-NOT: define {{.*}} spir_func void @func3_TU1()
; CHECK-TU1: define {{.*}} spir_func void @func3_TU1()
define dso_local spir_func void @func3_TU1() {
entry:
  ret void
}

; CHECK-TU0-TXT: TU0_kernel
; CHECK-TU1-TXT-NOT: TU0_kernel

; CHECK-TU0: define dso_local spir_kernel void @TU0_kernel()
; CHECK-TU1-NOT: define dso_local spir_kernel void @TU0_kernel()
define dso_local spir_kernel void @TU0_kernel() #1 {
entry:
; CHECK-TU0: call spir_func void @func_TU0()
  call spir_func void @func_TU0()
  ret void
}

; CHECK-TU0: define {{.*}} spir_func void @func_TU0()
; CHECK-TU1-NOT: define {{.*}} spir_func void @func_TU0()
define dso_local spir_func void @func_TU0() {
entry:
; CHECK-TU0: %0 = load i32, ptr addrspace(4) addrspacecast (ptr addrspace(1) @GV to ptr addrspace(4)), align 4
  %0 = load i32, ptr addrspace(4) getelementptr inbounds ([1 x i32], ptr addrspace(4) addrspacecast (ptr addrspace(1) @GV to ptr addrspace(4)), i64 0, i64 0), align 4
  ret void
}

attributes #0 = { "module-id"="TU1.cpp" }
attributes #1 = { "module-id"="TU2.cpp" }

; Metadata is saved in both modules.
; CHECK: !opencl.spir.version = !{!0, !0}
; CHECK: !spirv.Source = !{!1, !1}

!opencl.spir.version = !{!0, !0}
!spirv.Source = !{!1, !1}

; CHECK: !0 = !{i32 1, i32 2}
; CHECK: !1 = !{i32 4, i32 100000}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
