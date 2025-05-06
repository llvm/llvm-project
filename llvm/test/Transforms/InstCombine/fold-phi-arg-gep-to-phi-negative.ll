; RUN: opt < %s -passes=instcombine -S | FileCheck %s

%vec = type { %vec_base }
%vec_base = type { [4 x float] }
%foo = type { %vec, %vec}

define void @test(i1 %tobool, ptr addrspace(1) %add.ptr.i) {
entry:
  %lane.0 = alloca %foo, align 16
  %lane.15 = insertelement <16 x ptr> undef, ptr %lane.0, i64 0
  %mm_vectorGEP = getelementptr inbounds %foo, <16 x ptr> %lane.15, <16 x i64> zeroinitializer, <16 x i32> splat (i32 1), <16 x i32> zeroinitializer, <16 x i32> zeroinitializer, <16 x i64> splat (i64 1)
  %mm_vectorGEP2 = getelementptr inbounds %foo, <16 x ptr> %lane.15, <16 x i64> zeroinitializer, <16 x i32> zeroinitializer, <16 x i32> zeroinitializer, <16 x i32> zeroinitializer, <16 x i64> splat (i64 1)
  br i1 %tobool, label %f1, label %f0

f0:
; CHECK: f0:
; CHECK-NEXT: %mm_vectorGEP = getelementptr inbounds %foo, <16 x ptr> %lane.15, <16 x i64> zeroinitializer, <16 x i32> splat (i32 1), <16 x i32> zeroinitializer, <16 x i32> zeroinitializer, <16 x i64> splat (i64 1)
  br label %merge

f1:
; CHECK: f1:
; CHECK-NEXT: %mm_vectorGEP2 = getelementptr inbounds %foo, <16 x ptr> %lane.15, <16 x i64> zeroinitializer, <16 x i32> zeroinitializer, <16 x i32> zeroinitializer, <16 x i32> zeroinitializer, <16 x i64> splat (i64 1)
  br label %merge

merge:
; CHECK: merge:
; CHECK-NEXT: %vec.phi14 = phi <16 x ptr> [ %mm_vectorGEP, %f0 ], [ %mm_vectorGEP2, %f1 ]
  %vec.phi14 = phi <16 x ptr> [ %mm_vectorGEP, %f0], [ %mm_vectorGEP2, %f1 ]
  %wide.masked.gather15 = call <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr> %vec.phi14, i32 4, <16 x i1> splat (i1 true), <16 x float> poison)
  %wide.masked.gather15.extract.15. = extractelement <16 x float> %wide.masked.gather15, i32 15
  store float %wide.masked.gather15.extract.15., ptr addrspace(1) %add.ptr.i, align 4
  ret void
}

declare <16 x float> @llvm.masked.gather.v16f32.v16p0(<16 x ptr>, i32 immarg, <16 x i1>, <16 x float>)
