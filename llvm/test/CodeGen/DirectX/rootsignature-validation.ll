; RUN: opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 
; expected-no-diagnostics

;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind     Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ---------
; In                                texture  struct         r/o      T0             t0         1
; Out                                   UAV  struct         r/w      U0             u0         1
; UAV3                                  UAV  struct         r/w      U1             u1         1
; UAV1                                  UAV  struct         r/w      U2             u2         1
; UAV                                   UAV  struct         r/w      U3    u4294967294         1
; CB                                cbuffer      NA          NA     CB0     cb3,space1         1
;
; ModuleID = '../clang/test/SemaHLSL/RootSignature-Validation.hlsl'
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.5-unknown-shadermodel6.5-compute"

%__cblayout_CB = type <{ float }>
%"StructuredBuffer<int32_t>" = type { i32 }
%"RWStructuredBuffer<int32_t>" = type { i32 }
%"RWStructuredBuffer<float>" = type { float }
%CBuffer.CB = type { float }

@CB.cb = local_unnamed_addr global target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)) poison
@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1
@.str = private unnamed_addr constant [3 x i8] c"In\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"Out\00", align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"UAV\00", align 1
@.str.6 = private unnamed_addr constant [5 x i8] c"UAV1\00", align 1
@.str.10 = private unnamed_addr constant [5 x i8] c"UAV3\00", align 1
@In = external constant %"StructuredBuffer<int32_t>"
@Out = external constant %"RWStructuredBuffer<int32_t>"
@UAV3 = external constant %"RWStructuredBuffer<float>"
@UAV1 = external constant %"RWStructuredBuffer<float>"
@UAV = external constant %"RWStructuredBuffer<float>"
@CB = external constant %CBuffer.CB

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)) @llvm.dx.resource.handlefrombinding.tdx.CBuffer_tdx.Layout_s___cblayout_CBs_4_0tt(i32, i32, i32, i32, i1, ptr) #0

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define void @CSMain() local_unnamed_addr #1 {
entry:

  %CB = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)) @llvm.dx.resource.handlefrombinding(i32 1, i32 3, i32 1, i32 0, i1 false, ptr nonnull @CB.str)
  %Sampler = call target("dx.Sampler", 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 3, i32 1, i32 0, i1 false, ptr nonnull @Smp.str)
  %SB = tail call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @SB.str)
  %RWB =  tail call target("dx.RawBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @RWB.str)

  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32, i32, i32, i32, i1, ptr) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_1_0t(i32, i32, i32, i32, i1, ptr) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, i1, ptr) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", i32, 1, 0), i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0), i32) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i32) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_i32_0_0t(target("dx.RawBuffer", i32, 0, 0), i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare { float, i1 } @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.dx.resource.store.rawbuffer.tdx.RawBuffer_i32_1_0t.i32(target("dx.RawBuffer", i32, 1, 0), i32, i32, i32) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(read)
declare { float, float, float, float } @llvm.dx.resource.load.cbufferrow.4.f32.f32.f32.f32.tdx.CBuffer_tdx.Layout_s___cblayout_CBs_4_0tt(target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)), i32) #2

; uselistorder directives
uselistorder ptr @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t, { 2, 1, 0 }
uselistorder ptr @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_f32_1_0t, { 2, 1, 0 }

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "approx-func-fp-math"="false" "frame-pointer"="all" "hlsl.numthreads"="8,8,1" "hlsl.shader"="compute" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(write) }

!dx.rootsignatures = !{!0}
!llvm.module.flags = !{!9, !10}
!dx.valver = !{!11}
!llvm.ident = !{!12}
!dx.shaderModel = !{!13}
!dx.version = !{!14}
!dx.resources = !{!15}
!dx.entryPoints = !{!26}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3, !5, !7}
!2 = !{!"RootCBV", i32 0, i32 3, i32 1, i32 4}
!3 = !{!"DescriptorTable", i32 0, !4}
!4 = !{!"SRV", i32 1, i32 0, i32 0, i32 -1, i32 4}
!5 = !{!"DescriptorTable", i32 1, !6}
!6 = !{!"Sampler", i32 2, i32 0, i32 0, i32 -1, i32 0}
!7 = !{!"DescriptorTable", i32 0, !8}
!8 = !{!"UAV", i32 -1, i32 0, i32 0, i32 -1, i32 2}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{i32 1, i32 8}
!12 = !{!"clang version 21.0.0git (https://github.com/joaosaffran/llvm-project.git c16f15b4cd469a3f6efc2e4b0e098190d7fd0787)"}
!13 = !{!"cs", i32 6, i32 5}
!14 = !{i32 1, i32 5}
!15 = !{!16, !19, !24, null}
!16 = !{!17}
!17 = !{i32 0, ptr @In, !"In", i32 0, i32 0, i32 1, i32 12, i32 0, !18}
!18 = !{i32 1, i32 4}
!19 = !{!20, !21, !22, !23}
!20 = !{i32 0, ptr @Out, !"Out", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !18}
!21 = !{i32 1, ptr @UAV3, !"UAV3", i32 0, i32 1, i32 1, i32 12, i1 false, i1 false, i1 false, !18}
!22 = !{i32 2, ptr @UAV1, !"UAV1", i32 0, i32 2, i32 1, i32 12, i1 false, i1 false, i1 false, !18}
!23 = !{i32 3, ptr @UAV, !"UAV", i32 0, i32 -2, i32 1, i32 12, i1 false, i1 false, i1 false, !18}
!24 = !{!25}
!25 = !{i32 0, ptr @CB, !"CB", i32 1, i32 3, i32 1, i32 4, null}
!26 = !{ptr @CSMain, !"CSMain", null, !15, !27}
!27 = !{i32 0, i64 16, i32 4, !28}
!28 = !{i32 8, i32 8, i32 1}
