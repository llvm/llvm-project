; RUN: opt -S -passes='dxil-data-scalarization,function(scalarizer<load-store>),dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s -check-prefix DATACHECK
; RUN: opt -S -passes='dxil-data-scalarization,dxil-flatten-arrays,function(scalarizer<load-store>),dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library --filetype=asm -o - | FileCheck %s

; Make sure we don't touch arrays without vectors and that can recurse multiple-dimension arrays of vectors

@staticArray = internal global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4
@"groushared3dArrayofVectors" = local_unnamed_addr addrspace(3) global [3 x [3 x [3 x <4 x i32>]]] zeroinitializer, align 16

; CHECK @staticArray
; CHECK-NOT: @staticArray.scalarized
; CHECK-NOT: @staticArray.scalarized.1dim
; CHECK-NOT: @staticArray.1dim
; DATACHECK: @groushared3dArrayofVectors.scalarized = local_unnamed_addr addrspace(3) global [3 x [3 x [3 x [4 x i32]]]] zeroinitializer, align 16
; CHECK: @groushared3dArrayofVectors.scalarized.1dim = local_unnamed_addr addrspace(3) global [108 x i32] zeroinitializer, align 16
; DATACHECK-NOT: @groushared3dArrayofVectors
; CHECK-NOT: @groushared3dArrayofVectors.scalarized
