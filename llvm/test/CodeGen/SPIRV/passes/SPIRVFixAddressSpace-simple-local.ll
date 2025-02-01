; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3 %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After Fixup address space (spirv-fix-address-space) ***

@input = internal global i32 0
; CHECK: @input = internal addrspace(10) global i32 0
; CHECK: @main.local = internal addrspace(10) global i32 0

define spir_func i32 @foo(ptr noundef nonnull align 4 %param) {
; CHECK: define spir_func i32 @foo(ptr addrspace(10) noundef nonnull align 4 %param) {
  %1 = load i32, ptr %param, align 4
; CHECK:  %1 = load i32, ptr addrspace(10) %param, align 4
  ret i32 %1
; CHECK: ret i32 %1
}


define internal spir_func void @main() #0 {
; CHECK: define internal spir_func void @main() #0 {
entry:
; CHECK: entry:
  %0 = call token @llvm.experimental.convergence.entry()

  %1 = alloca i32
; CHECK-NOT: %1 = alloca

  %2 = load i32, ptr @input
; CHECK: %1 = load i32, ptr addrspace(10) @input
  store i32 %2, ptr %1
; CHECK: store i32 %1, ptr addrspace(10) @main.local

  %3 = call i32 @foo(ptr %1)
; CHECK: %2 = call i32 @foo(ptr addrspace(10) @main.local)
  %4 = load i32, ptr @input
; CHECK: %3 = load i32, ptr addrspace(10) @input

  ret void
; CHECK: ret void
}

attributes #0 = { convergent }

