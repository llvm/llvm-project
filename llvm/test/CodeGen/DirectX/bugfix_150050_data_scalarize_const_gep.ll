; RUN: opt -S -passes='dxil-data-scalarization' -mtriple=dxil-pc-shadermodel6.4-library %s | FileCheck %s --check-prefixes=SCHECK,CHECK
; RUN: opt -S -passes='dxil-data-scalarization,function(scalarizer<load-store>),dxil-flatten-arrays' -mtriple=dxil-pc-shadermodel6.4-library %s | FileCheck %s --check-prefixes=FCHECK,CHECK

@aTile = hidden addrspace(3) global [10 x [10 x <4 x i32>]] zeroinitializer, align 16
@bTile = hidden addrspace(3) global [10 x [10 x i32]] zeroinitializer, align 16
@cTile = internal global [2 x [2 x <2 x i32>]] zeroinitializer, align 16
@dTile = internal global [2 x [2 x [2 x <2 x i32>]]] zeroinitializer, align 16

define void @CSMain() {
; CHECK-LABEL: define void @CSMain() {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[AFRAGPACKED_I_SCALARIZE:%.*]] = alloca [4 x i32], align 16
;
; SCHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds [10 x <4 x i32>], ptr addrspace(3) getelementptr inbounds ([10 x [10 x [4 x i32]]], ptr addrspace(3) @aTile.scalarized, i32 0, i32 1), i32 0, i32 2
; SCHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr addrspace(3) [[TMP0]], align 16
; SCHECK-NEXT:    store <4 x i32> [[TMP1]], ptr [[AFRAGPACKED_I_SCALARIZE]], align 16
;
; FCHECK-NEXT:    [[AFRAGPACKED_I_SCALARIZE_I14:%.*]] = getelementptr [4 x i32], ptr [[AFRAGPACKED_I_SCALARIZE]], i32 0, i32 1
; FCHECK-NEXT:    [[AFRAGPACKED_I_SCALARIZE_I25:%.*]] = getelementptr [4 x i32], ptr [[AFRAGPACKED_I_SCALARIZE]], i32 0, i32 2
; FCHECK-NEXT:    [[AFRAGPACKED_I_SCALARIZE_I36:%.*]] = getelementptr [4 x i32], ptr [[AFRAGPACKED_I_SCALARIZE]], i32 0, i32 3
; FCHECK-NEXT:    [[DOTI07:%.*]] = load i32, ptr addrspace(3) getelementptr inbounds ([400 x i32], ptr addrspace(3) @aTile.scalarized.1dim, i32 0, i32 48), align 16
; FCHECK-NEXT:    [[DOTI119:%.*]] = load i32, ptr addrspace(3) getelementptr ([400 x i32], ptr addrspace(3) @aTile.scalarized.1dim, i32 0, i32 49), align 4
; FCHECK-NEXT:    [[DOTI2211:%.*]] = load i32, ptr addrspace(3) getelementptr ([400 x i32], ptr addrspace(3) @aTile.scalarized.1dim, i32 0, i32 50), align 8
; FCHECK-NEXT:    [[DOTI3313:%.*]] = load i32, ptr addrspace(3) getelementptr ([400 x i32], ptr addrspace(3) @aTile.scalarized.1dim, i32 0, i32 51), align 4
; FCHECK-NEXT:    store i32 [[DOTI07]], ptr [[AFRAGPACKED_I_SCALARIZE]], align 16
; FCHECK-NEXT:    store i32 [[DOTI119]], ptr [[AFRAGPACKED_I_SCALARIZE_I14]], align 4
; FCHECK-NEXT:    store i32 [[DOTI2211]], ptr [[AFRAGPACKED_I_SCALARIZE_I25]], align 8
; FCHECK-NEXT:    store i32 [[DOTI3313]], ptr [[AFRAGPACKED_I_SCALARIZE_I36]], align 4
;
; CHECK-NEXT:    ret void
entry:
  %aFragPacked.i = alloca <4 x i32>, align 16
  %0 = load <4 x i32>, ptr addrspace(3) getelementptr inbounds ([10 x <4 x i32>], ptr addrspace(3) getelementptr inbounds ([10 x [10 x <4 x i32>]], ptr addrspace(3) @aTile, i32 0, i32 1), i32 0, i32 2), align 16
  store <4 x i32> %0, ptr %aFragPacked.i, align 16
  ret void
}

define void @Main() {
; CHECK-LABEL: define void @Main() {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[BFRAGPACKED_I:%.*]] = alloca i32, align 16
;
; SCHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds [10 x i32], ptr addrspace(3) getelementptr inbounds ([10 x [10 x i32]], ptr addrspace(3) @bTile, i32 0, i32 1), i32 0, i32 1
; SCHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr addrspace(3) [[TMP0]], align 16
; SCHECK-NEXT:    store i32 [[TMP1]], ptr [[BFRAGPACKED_I]], align 16
;
; FCHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr addrspace(3) getelementptr inbounds ([100 x i32], ptr addrspace(3) @bTile.1dim, i32 0, i32 11), align 16
; FCHECK-NEXT:    store i32 [[TMP0]], ptr [[BFRAGPACKED_I]], align 16
;
; CHECK-NEXT:    ret void
entry:
  %bFragPacked.i = alloca i32, align 16
  %0 = load i32, ptr addrspace(3) getelementptr inbounds ([10 x i32], ptr addrspace(3) getelementptr inbounds ([10 x [10 x i32]], ptr addrspace(3) @bTile, i32 0, i32 1), i32 0, i32 1), align 16
  store i32 %0, ptr %bFragPacked.i, align 16
  ret void
}

define void @global_nested_geps_3d() {
; CHECK-LABEL: define void @global_nested_geps_3d() {
; SCHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds <2 x i32>, ptr getelementptr inbounds ([2 x <2 x i32>], ptr getelementptr inbounds ([2 x [2 x [2 x i32]]], ptr @cTile.scalarized, i32 0, i32 1), i32 0, i32 1), i32 0, i32 1
; SCHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[TMP1]], align 4
;
; FCHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr getelementptr inbounds ([8 x i32], ptr @cTile.scalarized.1dim, i32 0, i32 7), align 4
;
; CHECK-NEXT:    ret void
  %1 = load i32, i32* getelementptr inbounds (<2 x i32>, <2 x i32>* getelementptr inbounds ([2 x <2 x i32>], [2 x <2 x i32>]* getelementptr inbounds ([2 x [2 x <2 x i32>]], [2 x [2 x <2 x i32>]]* @cTile, i32 0, i32 1), i32 0, i32 1), i32 0, i32 1), align 4
  ret void
}

define void @global_nested_geps_4d() {
; CHECK-LABEL: define void @global_nested_geps_4d() {
; SCHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds <2 x i32>, ptr getelementptr inbounds ([2 x <2 x i32>], ptr getelementptr inbounds ([2 x [2 x <2 x i32>]], ptr getelementptr inbounds ([2 x [2 x [2 x [2 x i32]]]], ptr @dTile.scalarized, i32 0, i32 1), i32 0, i32 1), i32 0, i32 1), i32 0, i32 1
; SCHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[TMP1]], align 4
;
; FCHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr getelementptr inbounds ([16 x i32], ptr @dTile.scalarized.1dim, i32 0, i32 15), align 4
;
; CHECK-NEXT:    ret void
  %1 = load i32, i32* getelementptr inbounds (<2 x i32>, <2 x i32>* getelementptr inbounds ([2 x <2 x i32>], [2 x <2 x i32>]* getelementptr inbounds ([2 x [2 x <2 x i32>]], [2 x [2 x <2 x i32>]]* getelementptr inbounds ([2 x [2 x [2 x <2 x i32>]]], [2 x [2 x [2 x <2 x i32>]]]* @dTile, i32 0, i32 1), i32 0, i32 1), i32 0, i32 1), i32 0, i32 1), align 4
  ret void
}
