; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_EXT_relaxed_printf_string_address_space %s -o - | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK: OpExtension "SPV_EXT_relaxed_printf_string_address_space"
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] printf

; CHECK-ERROR: LLVM ERROR: SPV_EXT_relaxed_printf_string_address_space is required because printf uses a format string not in constant address space.

@.str = private unnamed_addr addrspace(1) constant [4 x i8] c"%d\0A\00", align 1

declare spir_func i32 @printf(ptr addrspace(4), ...)

define spir_kernel void @test_kernel() {
entry:
  ; Format string in addrspace(1) → cast to addrspace(4)
  %format = addrspacecast ptr addrspace(1) @.str to ptr addrspace(4)
  %val = alloca i32, align 4
  store i32 123, ptr %val, align 4
  %loaded = load i32, ptr %val, align 4

  ; Call printf with non-constant format string
  %call = call spir_func i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) %format, i32 %loaded)
  ret void
}
