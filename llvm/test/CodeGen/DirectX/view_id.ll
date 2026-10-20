; RUN: split-file %s %t
; RUN: opt -S -dxil-op-lower %t/vertex.ll | FileCheck %s
; RUN: opt -S -dxil-op-lower %t/pixel.ll | FileCheck %s
; RUN: not opt -S -dxil-op-lower %t/compute.ll 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -S -dxil-op-lower %t/sm60.ll 2>&1 | FileCheck %s --check-prefix=VERSION

; CHECK-LABEL: define i32 @main(
; CHECK: %[[VIEW:.*]] = call i32 @dx.op.viewID.i32(i32 138)
; CHECK: ret i32 %[[VIEW]]
; CHECK: declare i32 @dx.op.viewID.i32(i32) #[[#ATTR:]]
; CHECK: attributes #[[#ATTR]] = { nounwind memory(none) }

; ViewID is only available to graphics stages from shader model 6.1.
; STAGE: Cannot create ViewID operation: Invalid stage
; VERSION: Cannot create ViewID operation: No valid overloads for DXIL version 1.0

;--- vertex.ll
target triple = "dxil-pc-shadermodel6.1-vertex"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.view.id()
  ret i32 %0
}

;--- pixel.ll
target triple = "dxil-pc-shadermodel6.1-pixel"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.view.id()
  ret i32 %0
}

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
