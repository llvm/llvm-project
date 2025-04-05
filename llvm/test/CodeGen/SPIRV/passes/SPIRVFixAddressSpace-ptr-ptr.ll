; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3 %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After Fixup address space (spirv-fix-address-space) ***

%S = type { ptr }

@input = internal global i32 0
@struct = internal global %S { ptr @input }
; CHECK: @input = internal addrspace(10) global i32 0
; CHECK: @struct = internal addrspace(10) global %S { ptr addrspace(10) @input }


define internal spir_func i32 @main(i32 %index) #0 {
entry:
; CHECK: define internal spir_func i32 @main(i32 %index) #0 {
; CHECK: entry:

  %0 = getelementptr inbounds %S, ptr @struct, i32 0
; CHECK: %[[#pptr:]] = getelementptr inbounds %S, ptr addrspace(10) @struct, i32 0

  %1 = getelementptr ptr, ptr %0, i32 %index
; CHECK: %[[#ptr:]] = getelementptr ptr addrspace(10), ptr addrspace(10) %[[#pptr]], i32 %index

  %2 = load i32, ptr %1
; CHECK: %[[#load:]] = load i32, ptr addrspace(10) %[[#ptr]]

  ret i32 %2
; CHECK: ret i32 %[[#load]]
}

attributes #0 = { convergent }
