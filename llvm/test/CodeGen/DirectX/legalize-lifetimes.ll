; RUN: opt -S -passes='dxil-op-lower' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; CHECK-LABEL: define void @test_legal_lifetime() {
; CHECK-NEXT:    [[ACCUM_I_FLAT:%.*]] = alloca [1 x i32], align 4
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr i32, ptr [[ACCUM_I_FLAT]], i32 0
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 4, ptr nonnull [[ACCUM_I_FLAT]])
; CHECK-NEXT:    store i32 0, ptr [[GEP]], align 4
; CHECK-NEXT:    call void @llvm.lifetime.end.p0(i64 4, ptr nonnull [[ACCUM_I_FLAT]])
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
