;; No extension -> no LinkOnceODR
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-NOT: OpExtension "SPV_KHR_linkonce_odr"
; CHECK-SPIRV-NOT: OpDecorate %[[#]] LinkageAttributes "GV" LinkOnceODR 
; CHECK-SPIRV-NOT: OpDecorate %[[#]] LinkageAttributes "square" LinkOnceODR 

@GV = linkonce_odr addrspace(1) global [3 x i32] zeroinitializer, align 4

define spir_kernel void @k() {
entry:
  %call = call spir_func i32 @square(i32 2)
  ret void
}

define linkonce_odr dso_local spir_func i32 @square(i32 %in) {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, i32* %in.addr, align 4
  %0 = load i32, i32* %in.addr, align 4
  %1 = load i32, i32* %in.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}
