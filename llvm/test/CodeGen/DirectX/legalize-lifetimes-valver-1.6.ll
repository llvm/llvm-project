; RUN: opt -S -passes='dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,CHECK-SM63
; RUN: opt -S -passes='dxil-op-lower' -mtriple=dxil-pc-shadermodel6.6-library %s | FileCheck %s --check-prefixes=CHECK,CHECK-SM66
; RUN: opt -S -dxil-prepare -dxil-embed -mtriple=dxil-pc-shadermodel6.6-library %s | FileCheck %s --check-prefixes=CHECK,CHECK-EMBED

; Lifetime intrinsics are not valid prior to shader model 6.6 and are instead
; replaced with undef stores, provided the validator version is 1.6 or greater

; The dxil-embed pass will remove lifetime intrinsics because they transformed
; in a way that is illegal in modern LLVM IR before serializing to DXIL bitcode.
; So we check that no bitcast or lifetime intrinsics remain after dxil-embed

; CHECK-LABEL: define void @test_legal_lifetime() {
; CHECK-NEXT:       [[ACCUM_I_FLAT:%.*]] = alloca [1 x i32], align 4
; CHECK-NEXT:       [[GEP:%.*]] = getelementptr i32, ptr [[ACCUM_I_FLAT]], i32 0
; CHECK-SM63-NEXT:  store [1 x i32] undef, ptr [[ACCUM_I_FLAT]], align 4
; CHECK-SM66-NEXT:  call void @llvm.lifetime.start.p0(ptr nonnull [[ACCUM_I_FLAT]])
; CHECK-EMBED-NOT:  bitcast
; CHECK-EMBED-NOT:  lifetime
; CHECK-NEXT:       store i32 0, ptr [[GEP]], align 4
; CHECK-SM63-NEXT:  store [1 x i32] undef, ptr [[ACCUM_I_FLAT]], align 4
; CHECK-SM66-NEXT:  call void @llvm.lifetime.end.p0(ptr nonnull [[ACCUM_I_FLAT]])
; CHECK-EMBED-NOT:  bitcast
; CHECK-EMBED-NOT:  lifetime
; CHECK-NEXT:       ret void
;
define void @test_legal_lifetime()  {
  %accum.i.flat = alloca [1 x i32], align 4
  %gep = getelementptr i32, ptr %accum.i.flat, i32 0
  call void @llvm.lifetime.start.p0(ptr nonnull %accum.i.flat)
  store i32 0, ptr %gep, align 4
  call void @llvm.lifetime.end.p0(ptr nonnull %accum.i.flat)
  ret void
}

; Set the validator version to 1.6
!dx.valver = !{!0}
!0 = !{i32 1, i32 6}
