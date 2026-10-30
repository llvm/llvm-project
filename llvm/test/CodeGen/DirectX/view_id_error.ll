; RUN: split-file %s %t
; RUN: not opt -S -dxil-op-lower %t/compute.ll 2>&1 | FileCheck %s --check-prefix=CHECK-STAGE
; RUN: not opt -S -dxil-op-lower %t/sm60.ll 2>&1 | FileCheck %s --check-prefix=CHECK-VERSION

; ViewID is only available to graphics stages from shader model 6.1.
; CHECK-STAGE: Cannot create ViewID operation: Invalid stage
; CHECK-VERSION: Cannot create ViewID operation: No valid overloads for DXIL version 1.0

;--- compute.ll
target triple = "dxil-pc-shadermodel6.1-compute"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.view.id()
  ret i32 %0
}

;--- sm60.ll
target triple = "dxil-pc-shadermodel6.0-pixel"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.view.id()
  ret i32 %0
}
