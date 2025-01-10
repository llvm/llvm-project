; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3 %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After Fixup address space (spirv-fix-address-space) ***

@input = internal global i32 0
; CHECK: @input = internal addrspace(10) global i32 0

define spir_func i32 @foo(ptr noundef nonnull align 4 %param) #0 {
; CHECK: define spir_func i32 @foo(ptr addrspace(10) noundef nonnull align 4 %param) #0 {

  %1 = load i32, ptr %param, align 4
; CHECK:  %1 = load i32, ptr addrspace(10) %param, align 4
  ret i32 %1
}


define internal spir_func i32 @main() #1 {
; CHECK: define internal spir_func i32 @main() #1 {
entry:
; CHECK: entry:
  %0 = call token @llvm.experimental.convergence.entry()

  %1 = call i32 @foo(ptr @input)
; CHECK: %[[#call:]] = call i32 @foo(ptr addrspace(10) @input)

  ret i32 %1
; CHECK:   ret i32 %[[#call]]
}

attributes #0 = { alwaysinline }
attributes #1 = { convergent }
