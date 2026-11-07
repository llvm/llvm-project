; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: llc -mtriple=spirv-unknown-vulkan-compute -O0 %s -o - | FileCheck %s

; Regression test for the original bug: WavePrefixProduct on double4 triggers
; the structurizer to create routing blocks that violate SPIR-V merge dominance
; rules. This test exercises the full pattern with vector wave operations and
; interleaved conditionals, matching the real-world pattern from HLSL's
; WavePrefixProduct<double4>.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan-compute"

; CHECK: OpSelectionMerge
; CHECK: OpBranchConditional

define internal spir_func void @main() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %tid = call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %0) ]
  %cond1 = icmp ult i32 %tid, 16
  %cond2 = icmp ult i32 %tid, 8
  %cond3 = icmp ult i32 %tid, 4
  br i1 %cond1, label %scalar.wave, label %scalar.thread

scalar.wave:
  %s1 = call double @llvm.spv.wave.prefix.product.f64(double 1.0) [ "convergencectrl"(token %0) ]
  br i1 %cond2, label %scalar.wave2, label %scalar.thread2

scalar.wave2:
  %s2 = call double @llvm.spv.wave.prefix.product.f64(double 2.0) [ "convergencectrl"(token %0) ]
  br i1 %cond3, label %scalar.wave3, label %scalar.thread3

scalar.wave3:
  %s3 = call double @llvm.spv.wave.prefix.product.f64(double 3.0) [ "convergencectrl"(token %0) ]
  br label %vec.entry

scalar.thread3:
  br label %vec.entry

scalar.thread2:
  br label %vec.entry

scalar.thread:
  br label %vec.entry

vec.entry:
  %sv = phi double [ %s3, %scalar.wave3 ], [ 0.0, %scalar.thread3 ], [ 0.0, %scalar.thread2 ], [ 0.0, %scalar.thread ]
  br i1 %cond1, label %vec.wave, label %vec.thread

vec.wave:
  %v1 = call <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double> <double 1.0, double 1.0>) [ "convergencectrl"(token %0) ]
  br i1 %cond2, label %vec.wave2, label %vec.thread2

vec.wave2:
  %v2 = call <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double> <double 2.0, double 2.0>) [ "convergencectrl"(token %0) ]
  br i1 %cond3, label %vec.wave3, label %vec.thread3

vec.wave3:
  %v3 = call <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double> <double 3.0, double 3.0>) [ "convergencectrl"(token %0) ]
  br label %final

vec.thread3:
  br label %final

vec.thread2:
  br label %final

vec.thread:
  br label %final

final:
  %vr = phi <2 x double> [ %v3, %vec.wave3 ], [ zeroinitializer, %vec.thread3 ], [ zeroinitializer, %vec.thread2 ], [ zeroinitializer, %vec.thread ]
  ret void
}

declare token @llvm.experimental.convergence.entry() #1
declare i32 @__hlsl_wave_get_lane_index() #2
declare double @llvm.spv.wave.prefix.product.f64(double) #2
declare <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double>) #2

attributes #0 = { convergent noinline norecurse nounwind optnone "hlsl.numthreads"="32,1,1" "hlsl.shader"="compute" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn }
attributes #2 = { convergent nounwind }
