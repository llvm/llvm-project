; RUN: opt -S -passes='dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,CHECK-SM63
; RUN: opt -S -passes='dxil-op-lower' -mtriple=dxil-pc-shadermodel6.6-library %s | FileCheck %s --check-prefixes=CHECK,CHECK-SM66
; RUN: opt -S -dxil-op-lower -dxil-prepare -mtriple=dxil-pc-shadermodel6.6-library %s | FileCheck %s --check-prefixes=CHECK,CHECK-PREPARE

; CHECK-LABEL: define void @test_legal_lifetime() {
; 
; CHECK-SM63-NEXT:    [[ACCUM_I_FLAT:%.*]] = alloca [1 x i32], align 4
; CHECK-SM63-NEXT:    [[GEP:%.*]] = getelementptr i32, ptr [[ACCUM_I_FLAT]], i32 0
; CHECK-SM63-NEXT:    store [1 x i32] undef, ptr [[ACCUM_I_FLAT]], align 4
; CHECK-SM63-NEXT:    store i32 0, ptr [[GEP]], align 4
; CHECK-SM63-NEXT:    store [1 x i32] undef, ptr [[ACCUM_I_FLAT]], align 4
; 
; CHECK-SM66-NEXT:    [[ACCUM_I_FLAT:%.*]] = alloca [1 x i32], align 4
; CHECK-SM66-NEXT:    [[GEP:%.*]] = getelementptr i32, ptr [[ACCUM_I_FLAT]], i32 0
; CHECK-SM66-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[ACCUM_I_FLAT]])
; CHECK-SM66-NEXT:    store i32 0, ptr [[GEP]], align 4
; CHECK-SM66-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[ACCUM_I_FLAT]])
; 
; CHECK-PREPARE-NEXT:    [[ACCUM_I_FLAT:%.*]] = alloca [1 x i32], align 4
; CHECK-PREPARE-NEXT:    [[GEP:%.*]] = getelementptr i32, ptr [[ACCUM_I_FLAT]], i32 0
; CHECK-PREPARE-NEXT:    [[BITCAST:%.*]] = bitcast ptr [[ACCUM_I_FLAT]] to ptr
; CHECK-PREPARE-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[BITCAST]])
; CHECK-PREPARE-NEXT:    store i32 0, ptr [[GEP]], align 4
; CHECK-PREPARE-NEXT:    [[BITCAST:%.*]] = bitcast ptr [[ACCUM_I_FLAT]] to ptr
; CHECK-PREPARE-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[BITCAST]])
; 
; CHECK-NEXT:    ret void
;
define void @test_legal_lifetime()  {
  %accum.i.flat = alloca [1 x i32], align 4
  %gep = getelementptr i32, ptr %accum.i.flat, i32 0
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %accum.i.flat)
  store i32 0, ptr %gep, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %accum.i.flat)
  ret void
}

; CHECK-PREPARE-DAG: attributes [[LIFETIME_ATTRS:#.*]] = { nounwind }

; CHECK-PREPARE-DAG: ; Function Attrs: nounwind
; CHECK-PREPARE-DAG: declare void @llvm.lifetime.start.p0(i64, ptr) [[LIFETIME_ATTRS]]

; CHECK-PREPARE-DAG: ; Function Attrs: nounwind
; CHECK-PREPARE-DAG: declare void @llvm.lifetime.end.p0(i64, ptr) [[LIFETIME_ATTRS]]

; Function Attrs: nounwind memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64, ptr) #0

; Function Attrs: nounwind memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64, ptr) #0

attributes #0 = { nounwind memory(argmem: readwrite) }

; Set the validator version to 1.6
!dx.valver = !{!0}
!0 = !{i32 1, i32 6}
