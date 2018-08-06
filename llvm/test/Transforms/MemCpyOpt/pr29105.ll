; RUN: opt -memcpyopt -instcombine -S %s | FileCheck %s
%Foo = type { [2048 x i64] }

; Make sure that all mempcy calls are converted to memset calls, or removed.
; CHECK-LABEL: @baz(
; CHECK-NOT: call void @llvm.memcpy
define void @baz() unnamed_addr #0 {
entry-block:
  %x.sroa.0 = alloca [2048 x i64], align 8
  %tmp0 = alloca [2048 x i64], align 8
  %0 = bitcast [2048 x i64]* %tmp0 to i8*
  %tmp2 = alloca %Foo, align 8
  %x.sroa.0.0..sroa_cast6 = bitcast [2048 x i64]* %x.sroa.0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16384, i8* %x.sroa.0.0..sroa_cast6)
  call void @llvm.lifetime.start.p0i8(i64 16384, i8* %0)
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 16384, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %x.sroa.0.0..sroa_cast6, i8* align 8 %0, i64 16384, i1 false)
  call void @llvm.lifetime.end.p0i8(i64 16384, i8* %0)
  %1 = bitcast %Foo* %tmp2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16384, i8* %1)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %1, i8* align 8 %x.sroa.0.0..sroa_cast6, i64 16384, i1 false)
  call void @bar(%Foo* noalias nocapture nonnull dereferenceable(16384) %tmp2)
  call void @llvm.lifetime.end.p0i8(i64 16384, i8* %1)
  call void @llvm.lifetime.end.p0i8(i64 16384, i8* %x.sroa.0.0..sroa_cast6)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

declare void @bar(%Foo* noalias nocapture readonly dereferenceable(16384)) unnamed_addr #0

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #1

attributes #0 = { uwtable }
attributes #1 = { argmemonly nounwind }
