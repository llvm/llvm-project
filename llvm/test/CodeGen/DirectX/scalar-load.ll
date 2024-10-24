; RUN: opt -S -passes='dxil-data-scalarization,function(scalarizer<load-store>),dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library --filetype=asm -o - | FileCheck %s

; Make sure we can load groupshared, static vectors and arrays of vectors

@"arrayofVecData" = local_unnamed_addr addrspace(3) global [2 x <3 x float>] zeroinitializer, align 16
@"vecData" = external addrspace(3) global <4 x i32>, align 4
@staticArrayOfVecData = internal global [3 x <4 x i32>] [<4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> <i32 5, i32 6, i32 7, i32 8>, <4 x i32> <i32 9, i32 10, i32 11, i32 12>], align 4
@"groushared2dArrayofVectors" = local_unnamed_addr addrspace(3) global [3 x [ 3 x <4 x i32>]] zeroinitializer, align 16

; CHECK: @arrayofVecData.scalarized = local_unnamed_addr addrspace(3) global [2 x [3 x float]] zeroinitializer, align 16
; CHECK: @vecData.scalarized = external addrspace(3) global [4 x i32], align 4
; CHECK: @staticArrayOfVecData.scalarized = internal global [3 x [4 x i32]] {{\[}}[4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 5, i32 6, i32 7, i32 8], [4 x i32] [i32 9, i32 10, i32 11, i32 12]], align 4
; CHECK: @groushared2dArrayofVectors.scalarized = local_unnamed_addr addrspace(3) global [3 x [3 x [4 x i32]]] zeroinitializer, align 16

; CHECK-NOT: @arrayofVecData
; CHECK-NOT: @vecData
; CHECK-NOT: @staticArrayOfVecData
; CHECK-NOT: @groushared2dArrayofVectors


; CHECK-LABEL: load_array_vec_test
define <4 x i32> @load_array_vec_test() {
  ; CHECK-COUNT-8: load i32, ptr addrspace(3) {{(.*@arrayofVecData.scalarized.*|%.*)}}, align 4
  ; CHECK-NOT: load i32, ptr addrspace(3) {{.*}}, align 4
  %1 = load <4 x i32>, <4 x i32> addrspace(3)* getelementptr inbounds ([2 x <4 x i32>], [2 x <4 x i32>] addrspace(3)* @"arrayofVecData", i32 0, i32 0), align 4
  %2 = load <4 x i32>, <4 x i32> addrspace(3)* getelementptr inbounds ([2 x <4 x i32>], [2 x <4 x i32>] addrspace(3)* @"arrayofVecData", i32 0, i32 1), align 4
  %3 = add <4 x i32> %1, %2
  ret <4 x i32> %3
}

; CHECK-LABEL: load_vec_test
define <4 x i32> @load_vec_test() {
  ; CHECK-COUNT-4: load i32, ptr addrspace(3) {{(@vecData.scalarized|getelementptr \(i32, ptr addrspace\(3\) @vecData.scalarized, i32 .*\)|%.*)}}, align {{.*}}
  ; CHECK-NOT: load i32, ptr addrspace(3) {{.*}}, align 4 
  %1 = load <4 x i32>, <4 x i32> addrspace(3)* @"vecData", align 4
  ret <4 x i32> %1
}

; CHECK-LABEL: load_static_array_of_vec_test
define <4 x i32> @load_static_array_of_vec_test(i32 %index) {
  ; CHECK: getelementptr [3 x [4 x i32]], ptr @staticArrayOfVecData.scalarized, i32 0, i32 %index
  ; CHECK-COUNT-4: load i32, ptr {{.*}}, align 4
  ; CHECK-NOT: load i32, ptr {{.*}}, align 4
  %3 = getelementptr inbounds [3 x <4 x i32>], [3 x <4 x i32>]* @staticArrayOfVecData, i32 0, i32 %index
  %4 = load <4 x i32>, <4 x i32>* %3, align 4
  ret <4 x i32> %4
}

; CHECK-LABEL: multid_load_test
define <4 x i32> @multid_load_test()  {
  ; CHECK-COUNT-8: load i32, ptr addrspace(3) {{(.*@groushared2dArrayofVectors.scalarized.*|%.*)}}, align 4
  ; CHECK-NOT: load i32, ptr addrspace(3) {{.*}}, align 4
  %1 = load <4 x i32>, <4 x i32> addrspace(3)* getelementptr inbounds ([3 x [3 x <4 x i32>]], [3 x [3 x <4 x i32>]] addrspace(3)* @"groushared2dArrayofVectors", i32 0, i32 0, i32 0), align 4
  %2 = load <4 x i32>, <4 x i32> addrspace(3)* getelementptr inbounds ([3 x [3 x <4 x i32>]], [3 x [3 x <4 x i32>]] addrspace(3)* @"groushared2dArrayofVectors", i32 0, i32 1, i32 1), align 4
  %3 = add <4 x i32> %1, %2
  ret <4 x i32> %3
}
