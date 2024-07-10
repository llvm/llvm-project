; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa -debug 2>&1 | FileCheck %s --implicit-check-not="[root]"
; REQUIRES: asserts

; func_3 is never directly called, it needs to be considered
; as a root to handle this module correctly.

; CHECK:      [root] kernel_1
; CHECK-NEXT:   [dependency] func_1
; CHECK-NEXT:   [dependency] func_2
; CHECK-NEXT: [root] func_3
; CHECK-NEXT:   [dependency] func_2

define amdgpu_kernel void @kernel_1() {
entry:
  call void @func_1()
  ret void
}

define linkonce_odr hidden void @func_1() {
entry:
  %call = call i32 @func_2()
  ret void
}

define linkonce_odr hidden i32 @func_2() #0 {
entry:
  ret i32 0
}

define void @func_3() {
entry:
  %call = call i32 @func_2()
  ret void
}

attributes #0 = { noinline optnone }
