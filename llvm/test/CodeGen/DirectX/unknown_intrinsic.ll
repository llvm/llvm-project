; RUN: not opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: in function llvm.vector.reduce.and.v4i32 i32 (<4 x i32>): Unknown intrinsic?
define i32 @fn_and(<4 x i32> %0) local_unnamed_addr #0 {
  %2 = tail call i32 @llvm.vector.reduce.and.v4i32(<4 x i32> %0)
  ret i32 %2
}

declare i32 @llvm.vector.reduce.and.v4i32(<4 x i32>)

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
