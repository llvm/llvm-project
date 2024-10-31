; RUN: opt -S -passes='early-cse<memssa>' %s -o %t
; RUN: FileCheck --check-prefixes=CSE,CHECK %s < %t
; finish compiling to verify that dxil-op-lower removes the globals entirely
; RUN: opt -S -dxil-op-lower %t -o - | FileCheck --check-prefixes=LLC,CHECK %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.0-compute  --filetype=asm -o - %t | FileCheck --check-prefixes=LLC,CHECK %s
; RUN: llc -mtriple=dxil-pc-shadermodel6.6-compute  --filetype=asm -o - %t | FileCheck --check-prefixes=LLC,CHECK %s

; Ensure that EarlyCSE is able to eliminate unneeded loads of resource globals across typedBufferLoad.
; Also that DXILOpLowering eliminates the globals entirely.

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.6-unknown-shadermodel6.6-compute"

%"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", <4 x float>, 1, 0, 0) }

; LLC-NOT: @In = global
; LLC-NOT: @Out = global
@In = global %"class.hlsl::RWBuffer" zeroinitializer, align 4
@Out = global %"class.hlsl::RWBuffer" zeroinitializer, align 4

; Function Attrs: convergent noinline norecurse
; CHECK-LABEL define void @main()
define void @main() local_unnamed_addr #0 {
entry:
  ; LLC: %In_h.i1 = call %dx.types.Handle @dx.op.createHandle
  ; LLC: %Out_h.i2 = call %dx.types.Handle @dx.op.createHandle
  %In_h.i = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0) @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_1_0_0t(i32 0, i32 0, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %In_h.i, ptr @In, align 4
  %Out_h.i = call target("dx.TypedBuffer", <4 x float>, 1, 0, 0) @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_1_0_0t(i32 4, i32 1, i32 1, i32 0, i1 false)
  store target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %Out_h.i, ptr @Out, align 4
  ; CSE: call i32 @llvm.dx.flattened.thread.id.in.group()
  %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
  ; CHECK-NOT: load {{.*}} ptr @In
  %1 = load target("dx.TypedBuffer", <4 x float>, 1, 0, 0), ptr @In, align 4
  ; CSE: call noundef <4 x float> @llvm.dx.typedBufferLoad.v4f32.tdx.TypedBuffer_v4f32_1_0_0t
  %2 = call noundef <4 x float> @llvm.dx.typedBufferLoad.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %1, i32 %0)
  ; CHECK-NOT: load {{.*}} ptr @In
  %3 = load target("dx.TypedBuffer", <4 x float>, 1, 0, 0), ptr @In, align 4
  %4 = call noundef <4 x float> @llvm.dx.typedBufferLoad.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %3, i32 %0)
  %add.i = fadd <4 x float> %2, %4
  call void @llvm.dx.typedBufferStore.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0) %Out_h.i, i32 %0, <4 x float> %add.i)
  ; CHECK: ret void
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.dx.flattened.thread.id.in.group() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
; CSE: declare <4 x float> @llvm.dx.typedBufferLoad.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0), i32) [[ROAttr:#[0-9]+]]
declare <4 x float> @llvm.dx.typedBufferLoad.v4f32.tdx.TypedBuffer_v4f32_1_0_0t(target("dx.TypedBuffer", <4 x float>, 1, 0, 0), i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
; CSE: declare void @llvm.dx.typedBufferStore.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0), i32, <4 x float>) [[WOAttr:#[0-9]+]]
declare void @llvm.dx.typedBufferStore.tdx.TypedBuffer_v4f32_1_0_0t.v4f32(target("dx.TypedBuffer", <4 x float>, 1, 0, 0), i32, <4 x float>) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.TypedBuffer", <4 x float>, 1, 0, 0) @llvm.dx.handle.fromBinding.tdx.TypedBuffer_v4f32_1_0_0t(i32, i32, i32, i32, i1) #3

; CSE: attributes [[ROAttr]] = { {{.*}} memory(read) }
; CSE: attributes [[WOAttr]] = { {{.*}} memory(write) }

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!dx.valver = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 8}
!3 = !{!"clang version 20.0.0git (git@github.com:llvm/llvm-project.git 54dc966bd3d375d7c1604fac5fdac20989c1072a)"}
!4 = !{!5}
!5 = distinct !{!5, !6, !"_ZN4hlsl8RWBufferIDv4_fEixEi: %agg.result"}
!6 = distinct !{!6, !"_ZN4hlsl8RWBufferIDv4_fEixEi"}
!7 = !{!8, !9, i64 0}
!8 = !{!"_ZTSN4hlsl8RWBufferIDv4_fEE", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!12}
!12 = distinct !{!12, !13, !"_ZN4hlsl8RWBufferIDv4_fEixEi: %agg.result"}
!13 = distinct !{!13, !"_ZN4hlsl8RWBufferIDv4_fEixEi"}
!14 = !{!15}
!15 = distinct !{!15, !16, !"_ZN4hlsl8RWBufferIDv4_fEixEi: %agg.result"}
!16 = distinct !{!16, !"_ZN4hlsl8RWBufferIDv4_fEixEi"}
!17 = !{!18, !9, i64 0}
!18 = !{!"_ZTSN4hlsl8__detail18TypedResourceProxyIU9_Res_u_CTDv4_fu17__hlsl_resource_tS2_EE", !9, i64 0, !19, i64 4}
!19 = !{!"int", !9, i64 0}
!20 = !{!21}
!21 = distinct !{!21, !22, !"_ZN4hlsl8__detail18TypedResourceProxyIU9_Res_u_CTDv4_fu17__hlsl_resource_tS2_EaSES2_: %agg.result"}
!22 = distinct !{!22, !"_ZN4hlsl8__detail18TypedResourceProxyIU9_Res_u_CTDv4_fu17__hlsl_resource_tS2_EaSES2_"}
