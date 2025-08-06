RWBuffer<float> B[4][4] : register(u2);
RWBuffer<float> C[2][2][5] : register(u10, space1);

RWStructuredBuffer<float> Out;

[numthreads(4,1,1)]
void main() {
  Out[0] = B[3][2][0] + C[1][0][3][0];
}

// B -> 3 * 4 + 2 = 14
// C -> 1 * 2 * 5 + 0 * 5 + 3 = 13

/*
; ModuleID = 'test2-multi-dim-arrays.hlsl'
source_filename = "test2-multi-dim-arrays.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.6-unknown-shadermodel6.6-compute"

@.str = private unnamed_addr constant [2 x i8] c"B\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"C\00", align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"Out\00", align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @.str.4)
  %1 = tail call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 0, i32 2, i32 16, i32 14, i1 false, ptr nonnull @.str)
  %2 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_1_0_0t(target("dx.TypedBuffer", float, 1, 0, 0) %1, i32 0)
  %3 = load float, ptr %2, align 4, !tbaa !4
  %4 = tail call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 1, i32 10, i32 20, i32 13, i1 false, ptr nonnull @.str.2)
  %5 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_1_0_0t(target("dx.TypedBuffer", float, 1, 0, 0) %4, i32 0)
  %6 = load float, ptr %5, align 4, !tbaa !4
  %add.i = fadd reassoc nnan ninf nsz arcp afn float %6, %3
  %7 = tail call noundef nonnull align 4 dereferenceable(4) ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_1_0_0t(target("dx.TypedBuffer", float, 1, 0, 0) %0, i32 0)
  store float %add.i, ptr %7, align 4, !tbaa !4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefromimplicitbinding.tdx.TypedBuffer_f32_1_0_0t(i32, i32, i32, i32, i1, ptr) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_1_0_0t(target("dx.TypedBuffer", float, 1, 0, 0), i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32, i32, i32, i32, i1, ptr) #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "approx-func-fp-math"="true" "frame-pointer"="all" "hlsl.numthreads"="4,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!dx.valver = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 8}
!3 = !{!"clang version 21.0.0git (C:/llvm-project/clang 0c495ce9c4237f0f090b672efd94839e52cb5f18)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
*/
