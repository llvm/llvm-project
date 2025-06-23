; RUN: opt -S -passes='dxil-data-scalarization' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=SCHECK,CHECK
; RUN: opt -S -passes='dxil-data-scalarization,dxil-flatten-arrays,function(dxil-legalize)' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=LCHECK,CHECK

; SCHECK: @g.scalarized = local_unnamed_addr addrspace(3) global [2 x [2 x i32]] zeroinitializer, align 8
; LCHECK: @g.scalarized.1dim = local_unnamed_addr addrspace(3) global [4 x i32] zeroinitializer, align 8
@g = local_unnamed_addr addrspace(3) global [2 x <2 x i32>] zeroinitializer, align 8

; CHECK-LABEL: test_store
define void @test_store(<4 x float> noundef %a) #0 {
  ; SCHECK: store i32 0, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @g.scalarized, i32 12), align 4
  ; LCHECK: store i32 0, ptr addrspace(3) getelementptr inbounds nuw ([4 x i32], ptr addrspace(3) @g.scalarized.1dim, i32 0, i32 3), align 4
  store i32 0, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @g, i32 12), align 4
  ret void
} 

attributes #0 = { convergent norecurse nounwind "hlsl.export"}
