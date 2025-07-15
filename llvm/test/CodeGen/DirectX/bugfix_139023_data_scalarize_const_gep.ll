; RUN: opt -S -passes='dxil-data-scalarization' -mtriple=dxil-pc-shadermodel6.4-library %s | FileCheck %s --check-prefixes=SCHECK,CHECK
; RUN: opt -S -passes='dxil-data-scalarization,dxil-flatten-arrays' -mtriple=dxil-pc-shadermodel6.4-library %s | FileCheck %s --check-prefixes=FCHECK,CHECK

@aTile = hidden addrspace(3) global [10 x [10 x <4 x i32>]] zeroinitializer, align 16
@bTile = hidden addrspace(3) global [10 x [10 x i32]] zeroinitializer, align 16

define void @CSMain() {
; CHECK-LABEL: define void @CSMain() {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:  [[AFRAGPACKED_I_SCALARIZE:%.*]] = alloca [4 x i32], align 16
; SCHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds [10 x <4 x i32>], ptr addrspace(3) getelementptr inbounds ([10 x [10 x [4 x i32]]], ptr addrspace(3) @aTile.scalarized, i32 0, i32 1), i32 0, i32 2
; FCHECK-NEXT:    [[TMP0:%.*]] = load <4 x i32>, ptr addrspace(3) getelementptr inbounds ([400 x i32], ptr addrspace(3) @aTile.scalarized.1dim, i32 0, i32 48), align 16
; SCHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr addrspace(3) [[TMP0]], align 16
; SCHECK-NEXT:    store <4 x i32> [[TMP1]], ptr [[AFRAGPACKED_I_SCALARIZE]], align 16
; SCHECK-NEXT:    ret void
;
entry:
  %aFragPacked.i = alloca <4 x i32>, align 16
  %0 = load <4 x i32>, ptr addrspace(3) getelementptr inbounds ([10 x <4 x i32>], ptr addrspace(3) getelementptr inbounds ([10 x [10 x <4 x i32>]], ptr addrspace(3) @aTile, i32 0, i32 1), i32 0, i32 2), align 16
  store <4 x i32> %0, ptr %aFragPacked.i, align 16
  ret void
}

define void @Main() {
; CHECK-LABEL: define void @Main() {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:  [[BFRAGPACKED_I:%.*]] = alloca i32, align 16
; SCHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds [10 x i32], ptr addrspace(3) getelementptr inbounds ([10 x [10 x i32]], ptr addrspace(3) @bTile, i32 0, i32 1), i32 0, i32 2
; FCHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr addrspace(3) getelementptr inbounds ([100 x i32], ptr addrspace(3) @bTile.1dim, i32 0, i32 12), align 16
; SCHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr addrspace(3) [[TMP0]], align 16
; SCHECK-NEXT:    store i32 [[TMP1]], ptr [[BFRAGPACKED_I]], align 16
; SCHECK-NEXT:    ret void
;
entry:
  %bFragPacked.i = alloca i32, align 16
  %0 = load i32, ptr addrspace(3) getelementptr inbounds ([10 x i32], ptr addrspace(3) getelementptr inbounds ([10 x [10 x i32]], ptr addrspace(3) @bTile, i32 0, i32 1), i32 0, i32 2), align 16
  store i32 %0, ptr %bFragPacked.i, align 16
  ret void
}
