; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library --filetype=asm -o - | FileCheck %s

; Make sure we can store groupshared, static vectors and arrays of vectors

@"arrayofVecData" = local_unnamed_addr addrspace(3) global [2 x <3 x float>] zeroinitializer, align 16
@"vecData" = external addrspace(3) global <4 x i32>, align 4


; CHECK: @arrayofVecData.scalarized.1dim = local_unnamed_addr addrspace(3) global [6 x float] zeroinitializer, align 16
; CHECK: @vecData.scalarized = external addrspace(3) global [4 x i32], align 4
; CHECK-NOT: @arrayofVecData
; CHECK-NOT: @arrayofVecData.scalarized
; CHECK-NOT: @vecData

; CHECK-LABEL: store_array_vec_test
define void @store_array_vec_test () local_unnamed_addr #0 {
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [6 x float], ptr addrspace(3) @arrayofVecData.scalarized.1dim, i32 0, i32 0
; CHECK-NEXT:    store float 1.000000e+00, ptr addrspace(3) [[GEP]], align 16
; CHECK-NEXT:    store float 2.000000e+00, ptr addrspace(3) getelementptr ([6 x float], ptr addrspace(3) @arrayofVecData.scalarized.1dim, i32 0, i32 1), align 4
; CHECK-NEXT:    store float 3.000000e+00, ptr addrspace(3) getelementptr ([6 x float], ptr addrspace(3) @arrayofVecData.scalarized.1dim, i32 0, i32 2), align 8
; CHECK-NEXT:    store float 2.000000e+00, ptr addrspace(3) getelementptr inbounds ([6 x float], ptr addrspace(3) @arrayofVecData.scalarized.1dim, i32 0, i32 3), align 16
; CHECK-NEXT:    store float 4.000000e+00, ptr addrspace(3) getelementptr ([6 x float], ptr addrspace(3) @arrayofVecData.scalarized.1dim, i32 0, i32 4), align 4
; CHECK-NEXT:    store float 6.000000e+00, ptr addrspace(3) getelementptr ([6 x float], ptr addrspace(3) @arrayofVecData.scalarized.1dim, i32 0, i32 5), align 8
; CHECK-NEXT:    ret void
;
  store <3 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, ptr addrspace(3) @"arrayofVecData", align 16
  store <3 x float> <float 2.000000e+00, float 4.000000e+00, float 6.000000e+00>, ptr addrspace(3) getelementptr inbounds (i8, ptr addrspace(3) @"arrayofVecData", i32 12), align 16
  ret void
}

; CHECK-LABEL: store_vec_test
define void @store_vec_test(<4 x i32> %inputVec) #0 {
; CHECK-NEXT:    [[INPUTVEC_I01:%.*]] = extractelement <4 x i32> %inputVec, i32 0
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [4 x i32], ptr addrspace(3) @vecData.scalarized, i32 0, i32 0
; CHECK-NEXT:    store i32 [[INPUTVEC_I01]], ptr addrspace(3) [[GEP]], align 4
; CHECK-NEXT:    [[INPUTVEC_I12:%.*]] = extractelement <4 x i32> %inputVec, i32 1
; CHECK-NEXT:    store i32 [[INPUTVEC_I12]], ptr addrspace(3) getelementptr ([4 x i32], ptr addrspace(3) @vecData.scalarized, i32 0, i32 1), align 4
; CHECK-NEXT:    [[INPUTVEC_I23:%.*]] = extractelement <4 x i32> %inputVec, i32 2
; CHECK-NEXT:    store i32 [[INPUTVEC_I23]], ptr addrspace(3) getelementptr ([4 x i32], ptr addrspace(3) @vecData.scalarized, i32 0, i32 2), align 4
; CHECK-NEXT:    [[INPUTVEC_I34:%.*]] = extractelement <4 x i32> %inputVec, i32 3
; CHECK-NEXT:    store i32 [[INPUTVEC_I34]], ptr addrspace(3) getelementptr ([4 x i32], ptr addrspace(3) @vecData.scalarized, i32 0, i32 3), align 4
; CHECK-NEXT:    ret void
;
  store <4 x i32> %inputVec, <4 x i32> addrspace(3)* @"vecData", align 4
  ret void
}

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
