; RUN: not opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 
; CHECK: error: register srv (space=0, register=0) is bound to a texture or typed buffer.

;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind     Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ---------
; B                                 texture     f32         buf      T0             t0         1
; Out                                   UAV  struct         r/w      U0             u0         1
;
; ModuleID = '../clang/test/SemaHLSL/RootSignature-Validation-Textures.hlsl'
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.5-unknown-shadermodel6.5-compute"

%"Buffer<float>" = type { float }
%"RWStructuredBuffer<int32_t>" = type { i32 }

@.str = private unnamed_addr constant [4 x i8] c"Out\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"B\00", align 1
@B = external constant %"Buffer<float>"
@Out = external constant %"RWStructuredBuffer<int32_t>"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define void @CSMain() local_unnamed_addr #0 {
entry:
  %0 = tail call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @.str)
  %1 = tail call target("dx.TypedBuffer", float, 0, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_0_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @.str.2)
  %2 = call { float, i1 } @llvm.dx.resource.load.typedbuffer.f32.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0) %1, i32 0)
  %3 = extractvalue { float, i1 } %2, 0
  %conv.i = fptosi float %3 to i32
  call void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0) %0, i32 0, i32 0, i32 %conv.i)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32, i32, i32, i32, i1, ptr) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.TypedBuffer", float, 0, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_0_0_0t(i32, i32, i32, i32, i1, ptr) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0), i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0), i32) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare { float, i1 } @llvm.dx.resource.load.typedbuffer.f32.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0), i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0), i32, i32, i32) #3

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "approx-func-fp-math"="true" "frame-pointer"="all" "hlsl.numthreads"="8,8,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(write) }

!dx.rootsignatures = !{!0}
!llvm.module.flags = !{!4, !5}
!dx.valver = !{!6}
!llvm.ident = !{!7}
!dx.shaderModel = !{!8}
!dx.version = !{!9}
!dx.resources = !{!10}
!dx.entryPoints = !{!17}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3}
!2 = !{!"RootSRV", i32 0, i32 0, i32 0, i32 4}
!3 = !{!"RootUAV", i32 0, i32 0, i32 0, i32 2}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{i32 1, i32 8}
!7 = !{!"clang version 21.0.0git (https://github.com/joaosaffran/llvm-project.git c16f15b4cd469a3f6efc2e4b0e098190d7fd0787)"}
!8 = !{!"cs", i32 6, i32 5}
!9 = !{i32 1, i32 5}
!10 = !{!11, !14, null, null}
!11 = !{!12}
!12 = !{i32 0, ptr @B, !"B", i32 0, i32 0, i32 1, i32 10, i32 0, !13}
!13 = !{i32 0, i32 9}
!14 = !{!15}
!15 = !{i32 0, ptr @Out, !"Out", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !16}
!16 = !{i32 1, i32 4}
!17 = !{ptr @CSMain, !"CSMain", null, !10, !18}
!18 = !{i32 0, i64 16, i32 4, !19}
!19 = !{i32 8, i32 8, i32 1}
