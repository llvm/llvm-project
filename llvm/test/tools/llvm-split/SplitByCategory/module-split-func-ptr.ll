; This test checks that Module splitting can properly perform device code split by tracking
; all uses of functions (not only direct calls).

; RUN: llvm-split -split-by-category=module-id -S < %s -o %t
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix=CHECK-IR0
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefix=CHECK-IR1

; CHECK-IR0: define dso_local spir_kernel void @kernelA
;
; CHECK-IR1: @FuncTable = weak global ptr @func
; CHECK-IR1: define {{.*}} i32 @func
; CHECK-IR1: define weak_odr dso_local spir_kernel void @kernelB

@FuncTable = weak global ptr @func, align 8

define dso_local spir_func i32 @func(i32 %a) {
entry:
  ret i32 %a
}

define weak_odr dso_local spir_kernel void @kernelB() #0 {
entry:
  %0 = call i32 @indirect_call(ptr addrspace(4) addrspacecast ( ptr getelementptr inbounds ( [1 x ptr] , ptr @FuncTable, i64 0, i64 0) to ptr addrspace(4)), i32 0)
  ret void
}

define dso_local spir_kernel void @kernelA() #1 {
entry:
  ret void
}

declare dso_local spir_func i32 @indirect_call(ptr addrspace(4), i32) local_unnamed_addr

attributes #0 = { "module-id"="TU1.cpp" }
attributes #1 = { "module-id"="TU2.cpp" }

; CHECK: kernel1
; CHECK: kernel2
