; RUN: opt -S -passes='dxil-data-scalarization' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=SCHECK
; RUN: opt -S -passes='dxil-data-scalarization,dxil-flatten-arrays' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=FCHECK

; CHECK-LABEL: alloca_2d__vec_test
define void @alloca_2d__vec_test() local_unnamed_addr #2 {
  ; SCHECK:  alloca [2 x [4 x i32]], align 16
  ; FCHECK:  alloca [8 x i32], align 16
  %1 = alloca [2 x <4 x i32>], align 16
  ret void
}
