; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Switch lookup tables emitted by SimplifyCFG produce an i8 array global that
; is indexed with a runtime value. Ensure the global, GEP and load are all
; upcast to a DXIL-legal integer width and the trailing extend is folded away.

; CHECK: @switch.table.legalized = internal unnamed_addr constant [3 x i32] [i32 2, i32 3, i32 1], align 4
; CHECK: @signed.table.legalized = internal unnamed_addr constant [2 x i32] [i32 -1, i32 1], align 4

@switch.table = internal unnamed_addr constant [3 x i8] c"\02\03\01", align 4
@signed.table = internal unnamed_addr constant [2 x i8] c"\FF\01", align 4

define i32 @lookup_zext(i32 %idx) {
; CHECK-LABEL: define i32 @lookup_zext(
; CHECK-SAME: i32 [[IDX:%.*]]) {
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [3 x i32], ptr @switch.table.legalized, i32 0, i32 [[IDX]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
  %gep = getelementptr inbounds nuw [3 x i8], ptr @switch.table, i32 0, i32 %idx
  %load = load i8, ptr %gep, align 1
  %ext = zext i8 %load to i32
  ret i32 %ext
}

define i32 @lookup_sext(i32 %idx) {
; CHECK-LABEL: define i32 @lookup_sext(
; CHECK-SAME: i32 [[IDX:%.*]]) {
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [2 x i32], ptr @signed.table.legalized, i32 0, i32 [[IDX]]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
  %gep = getelementptr inbounds nuw [2 x i8], ptr @signed.table, i32 0, i32 %idx
  %load = load i8, ptr %gep, align 1
  %ext = sext i8 %load to i32
  ret i32 %ext
}
