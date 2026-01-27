; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpConstantTrue
; CHECK-SPIRV: OpConstantFalse

define spir_func zeroext i1 @f() {
entry:
  ret i1 true
}

define spir_func zeroext i1 @f2() {
entry:
  ret i1 false
}

define spir_kernel void @test(ptr addrspace(1) %i) {
entry:
  %i.addr = alloca ptr addrspace(1), align 4
  store ptr addrspace(1) %i, ptr %i.addr, align 4
  %call = call spir_func zeroext i1 @f()
  %conv = zext i1 %call to i32
  %0 = load ptr addrspace(1), ptr %i.addr, align 4
  store i32 %conv, ptr addrspace(1) %0, align 4
  ret void
}
