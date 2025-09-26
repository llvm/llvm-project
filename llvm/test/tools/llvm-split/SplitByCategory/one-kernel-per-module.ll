; Test checks "kernel" splitting mode.

; RUN: llvm-split -split-by-category=kernel -S < %s -o %t.files
; RUN: FileCheck %s -input-file=%t.files_0.ll --check-prefixes CHECK-MODULE0,CHECK
; RUN: FileCheck %s -input-file=%t.files_1.ll --check-prefixes CHECK-MODULE1,CHECK
; RUN: FileCheck %s -input-file=%t.files_2.ll --check-prefixes CHECK-MODULE2,CHECK

;CHECK-MODULE0: @GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4
;CHECK-MODULE1-NOT: @GV
;CHECK-MODULE2-NOT: @GV
@GV = internal addrspace(1) constant [1 x i32] [i32 42], align 4

; CHECK-MODULE0-NOT: define dso_local spir_kernel void @TU0_kernelA
; CHECK-MODULE1-NOT: define dso_local spir_kernel void @TU0_kernelA
; CHECK-MODULE2: define dso_local spir_kernel void @TU0_kernelA
define dso_local spir_kernel void @TU0_kernelA() #0 {
entry:
; CHECK-MODULE2: call spir_func void @foo()
  call spir_func void @foo()
  ret void
}

; CHECK-MODULE0-NOT: define {{.*}} spir_func void @foo()
; CHECK-MODULE1-NOT: define {{.*}} spir_func void @foo()
; CHECK-MODULE2: define {{.*}} spir_func void @foo()
define dso_local spir_func void @foo() {
entry:
; CHECK-MODULE2: call spir_func void @bar()
  call spir_func void @bar()
  ret void
}

; CHECK-MODULE0-NOT: define {{.*}} spir_func void @bar()
; CHECK-MODULE1-NOT: define {{.*}} spir_func void @bar()
; CHECK-MODULE2: define {{.*}} spir_func void @bar()
define linkonce_odr dso_local spir_func void @bar() {
entry:
  ret void
}

; CHECK-MODULE0-NOT: define dso_local spir_kernel void @TU0_kernelB()
; CHECK-MODULE1: define dso_local spir_kernel void @TU0_kernelB()
; CHECK-MODULE2-NOT: define dso_local spir_kernel void @TU0_kernelB()
define dso_local spir_kernel void @TU0_kernelB() #0 {
entry:
; CHECK-MODULE1: call spir_func void @foo1()
  call spir_func void @foo1()
  ret void
}

; CHECK-MODULE0-NOT: define {{.*}} spir_func void @foo1()
; CHECK-MODULE1: define {{.*}} spir_func void @foo1()
; CHECK-MODULE2-NOT: define {{.*}} spir_func void @foo1()
define dso_local spir_func void @foo1() {
entry:
  ret void
}

; CHECK-MODULE0: define dso_local spir_kernel void @TU1_kernel()
; CHECK-MODULE1-NOT: define dso_local spir_kernel void @TU1_kernel()
; CHECK-MODULE2-NOT: define dso_local spir_kernel void @TU1_kernel()
define dso_local spir_kernel void @TU1_kernel() #1 {
entry:
; CHECK-MODULE0: call spir_func void @foo2()
  call spir_func void @foo2()
  ret void
}

; CHECK-MODULE0: define {{.*}} spir_func void @foo2()
; CHECK-MODULE1-NOT: define {{.*}} spir_func void @foo2()
; CHECK-MODULE2-NOT: define {{.*}} spir_func void @foo2()
define dso_local spir_func void @foo2() {
entry:
; CHECK-MODULE0: %0 = load i32, ptr addrspace(4) addrspacecast (ptr addrspace(1) @GV to ptr addrspace(4)), align 4
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

; CHECK; !0 = !{i32 1, i32 2}
; CHECK; !1 = !{i32 4, i32 100000}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
