; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3 %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After Fixup address space (spirv-fix-address-space) ***

@input = internal global i32 0
; CHECK: @input = internal addrspace(10) global i32 0

define spir_func void @foo(ptr noundef nonnull align 4 %param) {
; CHECK: define spir_func void @foo(ptr addrspace(10) noundef nonnull align 4 %param) {

  %1 = load ptr, ptr %param, align 4
; CHECK:  %1 = load ptr addrspace(10), ptr addrspace(10) %param, align 4

  ret void
}


define internal spir_func void @main() #0 {
; CHECK: define internal spir_func void @main() #0 {
entry:
; CHECK: entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %v = load i32, ptr @input
; CHECK:   %v = load i32, ptr addrspace(10) @input

  ret void
; CHECK:   ret void
}

attributes #0 = { convergent }
