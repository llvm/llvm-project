; RUN: opt -S -passes='dxil-post-optimization-validation' -mtriple=dxil-pc-shadermodel6.6-compute %s 2>&1 
; expected-no-diagnostics


; Root Signature(
;   CBV(b3, space=1, visibility=SHADER_VISIBILITY_ALL)
;   DescriptorTable(SRV(t0, space=0, numDescriptors=1), visibility=SHADER_VISIBILITY_ALL)
;   DescriptorTable(Sampler(s0, numDescriptors=2), visibility=SHADER_VISIBILITY_VERTEX)
;   DescriptorTable(UAV(u0, numDescriptors=unbounded), visibility=SHADER_VISIBILITY_ALL)

%__cblayout_CB = type <{ float }>

@CB.str = private unnamed_addr constant [3 x i8] c"CB\00", align 1
@Smp.str = private unnamed_addr constant [4 x i8] c"Smp\00", align 1
@SB.str = private unnamed_addr constant [3 x i8] c"SB\00", align 1
@RWB.str = private unnamed_addr constant [4 x i8] c"RWB\00", align 1

define void @CSMain() "hlsl.shader"="compute" {
entry:

  %CB = tail call target("dx.CBuffer", target("dx.Layout", %__cblayout_CB, 4, 0)) @llvm.dx.resource.handlefrombinding(i32 1, i32 3, i32 1, i32 0, i1 false, ptr nonnull @CB.str)
  %Sampler = call target("dx.Sampler", 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 3, i32 1, i32 0, i1 false, ptr nonnull @Smp.str)
  %SB = tail call target("dx.RawBuffer", i32, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i32_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @SB.str)
  %RWB =  tail call target("dx.RawBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr nonnull @RWB.str)

  ret void
}

!dx.rootsignatures = !{!0}

!0 = !{ptr @CSMain, !1, i32 2}
!1 = !{!2, !3, !5, !7}
!2 = !{!"RootCBV", i32 0, i32 3, i32 1, i32 4}
!3 = !{!"DescriptorTable", i32 0, !4}
!4 = !{!"SRV", i32 1, i32 0, i32 0, i32 -1, i32 0}
!5 = !{!"DescriptorTable", i32 0, !6}
!6 = !{!"Sampler", i32 5, i32 3, i32 2, i32 -1, i32 0}
!7 = !{!"DescriptorTable", i32 0, !8}
!8 = !{!"UAV", i32 -1, i32 0, i32 0, i32 -1, i32 2}
