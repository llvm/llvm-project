; RUN: opt -S -passes='dxil-data-scalarization,scalarizer<load-store>,dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure we can store groupshared, static vectors and arrays of vectors

@"arrayofVecData" = local_unnamed_addr addrspace(3) global [2 x <3 x float>] zeroinitializer, align 16
@"vecData" = external addrspace(3) global <4 x i32>, align 4

; CHECK: @arrayofVecData.scalarized = local_unnamed_addr addrspace(3) global [2 x [3 x float]] zeroinitializer, align 16
; CHECK: @vecData.scalarized = external addrspace(3) global [4 x i32], align 4
; CHECK-NOT: @arrayofVecData
; CHECK-NOT: @vecData

; CHECK-LABEL: store_array_vec_test
define void @store_array_vec_test () local_unnamed_addr #0 {
    ; CHECK-COUNT-6: store float {{1|2|3|4|6}}.000000e+00, ptr addrspace(3) {{(.*@arrayofVecData.scalarized.*|%.*)}}, align {{4|8|16}}
    ; CHECK-NEXT: ret void
    store <3 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, ptr addrspace(3) @"arrayofVecData", align 16 
    store <3 x float> <float 2.000000e+00, float 4.000000e+00, float 6.000000e+00>, ptr addrspace(3)   getelementptr inbounds (i8, ptr addrspace(3) @"arrayofVecData", i32 16), align 16 
    ret void
 } 

; CHECK-LABEL: store_vec_test
define void @store_vec_test(<4 x i32> %inputVec) #0 {
  ; CHECK-COUNT-4: store i32 %inputVec.{{.*}}, ptr addrspace(3) {{(@vecData.scalarized|getelementptr \(i32, ptr addrspace\(3\) @vecData.scalarized, i32 .*\)|%.*)}}, align 4 
  ; CHECK-NEXT: ret void
  store <4 x i32> %inputVec, <4 x i32> addrspace(3)* @"vecData", align 4
  ret void
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
