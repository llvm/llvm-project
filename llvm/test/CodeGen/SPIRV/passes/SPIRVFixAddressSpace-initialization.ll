; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3 %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After Fixup address space (spirv-fix-address-space) ***

@scalar = internal global i32 10
; CHECK: @scalar = internal addrspace(10) global i32 10

@pointer = internal global ptr null
; CHECK: @pointer = internal addrspace(10) global ptr addrspace(10) null

@array = internal global [2 x i32] [i32 1, i32 2]
; CHECK: @array = internal addrspace(10) global [2 x i32] [i32 1, i32 2]

@aggregate = internal global { i32, i32, ptr } { i32 3, i32 4, ptr null }
; CHECK: @aggregate = internal addrspace(10) global { i32, i32, ptr addrspace(10) } { i32 3, i32 4, ptr addrspace(10) null }



define internal spir_func i32 @main() #0 {
entry:
  ret i32 0
}

attributes #0 = { convergent }

