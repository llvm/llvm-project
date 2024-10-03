; RUN: opt -S -dxil-data-scalarization -scalarizer -scalarize-load-store -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library --filetype=asm -o - | FileCheck %s

; Make sure we don't touch arrays without vectors and that can recurse multiple-dimension arrays of vectors

@staticArray = internal global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4
@"groushared3dArrayofVectors" = local_unnamed_addr addrspace(3) global [3 x [3 x [3 x <4 x i32>]]] zeroinitializer, align 16

; CHECK @staticArray
; CHECK-NOT: @staticArray.scalarized
; CHECK: @groushared3dArrayofVectors.scalarized = local_unnamed_addr addrspace(3) global [3 x [3 x [3 x [4 x i32]]]] zeroinitializer, align 16
; CHECK-NOT: @groushared3dArrayofVectors
