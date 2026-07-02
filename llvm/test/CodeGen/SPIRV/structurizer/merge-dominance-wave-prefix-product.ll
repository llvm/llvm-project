; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
;
; Regression test for SPIR-V structurizer merge dominance bug.
; WavePrefixProduct on double4 creates nested convergent operations that
; trigger routing blocks with invalid merge assignments after structurization.
; This test verifies the fix produces valid SPIR-V.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv-unknown-vulkan-compute"

@.str = private unnamed_addr constant [3 x i8] c"In\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"Out1\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"Out2\00", align 1
@.str.6 = private unnamed_addr constant [5 x i8] c"Out3\00", align 1
@.str.8 = private unnamed_addr constant [5 x i8] c"Out4\00", align 1
@.str.10 = private unnamed_addr constant [5 x i8] c"Out5\00", align 1

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #0

; Function Attrs: convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none, target_mem: none)
define void @main() local_unnamed_addr #1 {
entry:
  %scalars.i = alloca [4 x double], align 8
  %vec2s.i = alloca [4 x <2 x double>], align 8
  %vec3s.i = alloca [4 x <3 x double>], align 8
  %vec4s.i = alloca [4 x <4 x double>], align 8
  %0 = tail call token @llvm.experimental.convergence.entry()
  %1 = tail call target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f64_12_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %2 = tail call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %3 = tail call noundef align 8 dereferenceable(32) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f64_12_0t.i32(target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 0) %1, i32 0)
  %4 = load <4 x double>, ptr addrspace(11) %3, align 8, !tbaa !6
  %cmp.i = icmp eq i32 %2, 0
  br i1 %cmp.i, label %cond.end.i.thread, label %cond.end.i

cond.end.i.thread:                                ; preds = %entry
  %5 = extractelement <4 x double> %4, i64 0
  %hlsl.wave.prefix.product2.i = call reassoc nnan ninf nsz arcp afn double @llvm.spv.wave.prefix.product.f64(double %5) [ "convergencectrl"(token %0) ]
  br label %cond.end7.i.thread

cond.end.i:                                       ; preds = %entry
  %cmp3.i = icmp eq i32 %2, 1
  br i1 %cmp3.i, label %cond.end.i.cond.end7.i.thread_crit_edge, label %cond.end7.i

cond.end.i.cond.end7.i.thread_crit_edge:          ; preds = %cond.end.i
  %.pre = extractelement <4 x double> %4, i64 0
  br label %cond.end7.i.thread

cond.end7.i.thread:                               ; preds = %cond.end.i.cond.end7.i.thread_crit_edge, %cond.end.i.thread
  %.pre-phi = phi double [ %.pre, %cond.end.i.cond.end7.i.thread_crit_edge ], [ %5, %cond.end.i.thread ]
  %cond.i4 = phi double [ 0.000000e+00, %cond.end.i.cond.end7.i.thread_crit_edge ], [ %hlsl.wave.prefix.product2.i, %cond.end.i.thread ]
  %hlsl.wave.prefix.product5.i = call reassoc nnan ninf nsz arcp afn double @llvm.spv.wave.prefix.product.f64(double %.pre-phi) [ "convergencectrl"(token %0) ]
  br label %cond.end13.i.thread

cond.end7.i:                                      ; preds = %cond.end.i
  %cmp9.i = icmp ult i32 %2, 3
  br i1 %cmp9.i, label %cond.end7.i.cond.end13.i.thread_crit_edge, label %cond.end13.i

cond.end7.i.cond.end13.i.thread_crit_edge:        ; preds = %cond.end7.i
  %.pre439 = extractelement <4 x double> %4, i64 0
  br label %cond.end13.i.thread

cond.end13.i.thread:                              ; preds = %cond.end7.i.cond.end13.i.thread_crit_edge, %cond.end7.i.thread
  %.pre-phi440 = phi double [ %.pre439, %cond.end7.i.cond.end13.i.thread_crit_edge ], [ %.pre-phi, %cond.end7.i.thread ]
  %cond8.i16 = phi double [ 0.000000e+00, %cond.end7.i.cond.end13.i.thread_crit_edge ], [ %hlsl.wave.prefix.product5.i, %cond.end7.i.thread ]
  %cond.i314 = phi double [ 0.000000e+00, %cond.end7.i.cond.end13.i.thread_crit_edge ], [ %cond.i4, %cond.end7.i.thread ]
  %cmp3.i512 = phi i1 [ false, %cond.end7.i.cond.end13.i.thread_crit_edge ], [ true, %cond.end7.i.thread ]
  %hlsl.wave.prefix.product11.i = call reassoc nnan ninf nsz arcp afn double @llvm.spv.wave.prefix.product.f64(double %.pre-phi440) [ "convergencectrl"(token %0) ]
  br label %cond.end19.i

cond.end13.i:                                     ; preds = %cond.end7.i
  %cmp15.i = icmp eq i32 %2, 3
  br i1 %cmp15.i, label %cond.end13.i.cond.end19.i_crit_edge, label %_Z4mainDv3_j.exit

cond.end13.i.cond.end19.i_crit_edge:              ; preds = %cond.end13.i
  %.pre441 = extractelement <4 x double> %4, i64 0
  br label %cond.end19.i

cond.end19.i:                                     ; preds = %cond.end13.i.cond.end19.i_crit_edge, %cond.end13.i.thread
  %.pre-phi442 = phi double [ %.pre441, %cond.end13.i.cond.end19.i_crit_edge ], [ %.pre-phi440, %cond.end13.i.thread ]
  %cond14.i34 = phi double [ 0.000000e+00, %cond.end13.i.cond.end19.i_crit_edge ], [ %hlsl.wave.prefix.product11.i, %cond.end13.i.thread ]
  %cmp3.i51132 = phi i1 [ false, %cond.end13.i.cond.end19.i_crit_edge ], [ %cmp3.i512, %cond.end13.i.thread ]
  %cond.i31330 = phi double [ 0.000000e+00, %cond.end13.i.cond.end19.i_crit_edge ], [ %cond.i314, %cond.end13.i.thread ]
  %cond8.i1528 = phi double [ 0.000000e+00, %cond.end13.i.cond.end19.i_crit_edge ], [ %cond8.i16, %cond.end13.i.thread ]
  %cmp9.i1726 = phi i1 [ false, %cond.end13.i.cond.end19.i_crit_edge ], [ true, %cond.end13.i.thread ]
  %hlsl.wave.prefix.product17.i = call reassoc nnan ninf nsz arcp afn double @llvm.spv.wave.prefix.product.f64(double %.pre-phi442) [ "convergencectrl"(token %0) ]
  br i1 %cmp.i, label %cond.true22.i, label %cond.end25.i

cond.true22.i:                                    ; preds = %cond.end19.i
  %6 = shufflevector <4 x double> %4, <4 x double> poison, <2 x i32> <i32 0, i32 1>
  %hlsl.wave.prefix.product23.i = call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double> %6) [ "convergencectrl"(token %0) ]
  br i1 %cmp3.i51132, label %cond.true28.i, label %cond.end31.i

cond.end25.i:                                     ; preds = %cond.end19.i
  br i1 %cmp3.i51132, label %cond.end25.i.cond.true28.i_crit_edge, label %cond.end31.i

cond.end25.i.cond.true28.i_crit_edge:             ; preds = %cond.end25.i
  %.pre443 = shufflevector <4 x double> %4, <4 x double> poison, <2 x i32> <i32 0, i32 1>
  br label %cond.true28.i

cond.true28.i:                                    ; preds = %cond.end25.i.cond.true28.i_crit_edge, %cond.true22.i
  %.pre-phi444 = phi <2 x double> [ %.pre443, %cond.end25.i.cond.true28.i_crit_edge ], [ %6, %cond.true22.i ]
  %cond26.i74 = phi <2 x double> [ zeroinitializer, %cond.end25.i.cond.true28.i_crit_edge ], [ %hlsl.wave.prefix.product23.i, %cond.true22.i ]
  %hlsl.wave.prefix.product29.i = call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double> %.pre-phi444) [ "convergencectrl"(token %0) ]
  br i1 %cmp9.i1726, label %cond.true34.i, label %cond.true40.i

cond.end31.i:                                     ; preds = %cond.true22.i, %cond.end25.i
  %cond26.i66 = phi <2 x double> [ %hlsl.wave.prefix.product23.i, %cond.true22.i ], [ zeroinitializer, %cond.end25.i ]
  %.pre445 = shufflevector <4 x double> %4, <4 x double> poison, <2 x i32> <i32 0, i32 1>
  br i1 %cmp9.i1726, label %cond.true34.i, label %cond.true40.i

cond.true34.i:                                    ; preds = %cond.end31.i, %cond.true28.i
  %.pre-phi446 = phi <2 x double> [ %.pre-phi444, %cond.true28.i ], [ %.pre445, %cond.end31.i ]
  %cond32.i101 = phi <2 x double> [ %hlsl.wave.prefix.product29.i, %cond.true28.i ], [ zeroinitializer, %cond.end31.i ]
  %cmp3.i51131466396 = phi i1 [ true, %cond.true28.i ], [ false, %cond.end31.i ]
  %cond26.i6693 = phi <2 x double> [ %cond26.i74, %cond.true28.i ], [ %cond26.i66, %cond.end31.i ]
  %hlsl.wave.prefix.product35.i = call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double> %.pre-phi446) [ "convergencectrl"(token %0) ]
  br label %cond.true40.i

cond.true40.i:                                    ; preds = %cond.end31.i, %cond.true28.i, %cond.true34.i
  %.pre-phi448 = phi <2 x double> [ %.pre-phi446, %cond.true34.i ], [ %.pre-phi444, %cond.true28.i ], [ %.pre445, %cond.end31.i ]
  %cond38.i131 = phi <2 x double> [ %hlsl.wave.prefix.product35.i, %cond.true34.i ], [ zeroinitializer, %cond.true28.i ], [ zeroinitializer, %cond.end31.i ]
  %cond26.i6684130 = phi <2 x double> [ %cond26.i6693, %cond.true34.i ], [ %cond26.i74, %cond.true28.i ], [ %cond26.i66, %cond.end31.i ]
  %cmp3.i51131466387127 = phi i1 [ %cmp3.i51131466396, %cond.true34.i ], [ true, %cond.true28.i ], [ false, %cond.end31.i ]
  %cmp9.i1725496090124 = phi i1 [ true, %cond.true34.i ], [ false, %cond.true28.i ], [ false, %cond.end31.i ]
  %cond32.i92122 = phi <2 x double> [ %cond32.i101, %cond.true34.i ], [ %hlsl.wave.prefix.product29.i, %cond.true28.i ], [ zeroinitializer, %cond.end31.i ]
  %hlsl.wave.prefix.product41.i = call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double> %.pre-phi448) [ "convergencectrl"(token %0) ]
  br i1 %cmp.i, label %cond.true46.i, label %cond.end49.i

cond.true46.i:                                    ; preds = %cond.true40.i
  %7 = shufflevector <4 x double> %4, <4 x double> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %hlsl.wave.prefix.product47.i = call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.spv.wave.prefix.product.v3f64(<3 x double> %7) [ "convergencectrl"(token %0) ]
  br i1 %cmp3.i51131466387127, label %cond.true52.i, label %cond.end55.i

cond.end49.i:                                     ; preds = %cond.true40.i
  br i1 %cmp3.i51131466387127, label %cond.end49.i.cond.true52.i_crit_edge, label %cond.end55.i

cond.end49.i.cond.true52.i_crit_edge:             ; preds = %cond.end49.i
  %.pre449 = shufflevector <4 x double> %4, <4 x double> poison, <3 x i32> <i32 0, i32 1, i32 2>
  br label %cond.true52.i

cond.true52.i:                                    ; preds = %cond.end49.i.cond.true52.i_crit_edge, %cond.true46.i
  %.pre-phi450 = phi <3 x double> [ %.pre449, %cond.end49.i.cond.true52.i_crit_edge ], [ %7, %cond.true46.i ]
  %cond50.i189 = phi <3 x double> [ zeroinitializer, %cond.end49.i.cond.true52.i_crit_edge ], [ %hlsl.wave.prefix.product47.i, %cond.true46.i ]
  %hlsl.wave.prefix.product53.i = call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.spv.wave.prefix.product.v3f64(<3 x double> %.pre-phi450) [ "convergencectrl"(token %0) ]
  br i1 %cmp9.i1725496090124, label %cond.true58.i, label %cond.true64.i

cond.end55.i:                                     ; preds = %cond.true46.i, %cond.end49.i
  %cond50.i177 = phi <3 x double> [ %hlsl.wave.prefix.product47.i, %cond.true46.i ], [ zeroinitializer, %cond.end49.i ]
  %.pre451 = shufflevector <4 x double> %4, <4 x double> poison, <3 x i32> <i32 0, i32 1, i32 2>
  br i1 %cmp9.i1725496090124, label %cond.true58.i, label %cond.true64.i

cond.true58.i:                                    ; preds = %cond.end55.i, %cond.true52.i
  %.pre-phi452 = phi <3 x double> [ %.pre-phi450, %cond.true52.i ], [ %.pre451, %cond.end55.i ]
  %cond56.i228 = phi <3 x double> [ %hlsl.wave.prefix.product53.i, %cond.true52.i ], [ zeroinitializer, %cond.end55.i ]
  %cmp3.i51131466387117147172221 = phi i1 [ true, %cond.true52.i ], [ false, %cond.end55.i ]
  %cond50.i177216 = phi <3 x double> [ %cond50.i189, %cond.true52.i ], [ %cond50.i177, %cond.end55.i ]
  %hlsl.wave.prefix.product59.i = call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.spv.wave.prefix.product.v3f64(<3 x double> %.pre-phi452) [ "convergencectrl"(token %0) ]
  br label %cond.true64.i

cond.true64.i:                                    ; preds = %cond.end55.i, %cond.true52.i, %cond.true58.i
  %.pre-phi454 = phi <3 x double> [ %.pre-phi452, %cond.true58.i ], [ %.pre-phi450, %cond.true52.i ], [ %.pre451, %cond.end55.i ]
  %cond62.i270 = phi <3 x double> [ %hlsl.wave.prefix.product59.i, %cond.true58.i ], [ zeroinitializer, %cond.true52.i ], [ zeroinitializer, %cond.end55.i ]
  %cond50.i177203269 = phi <3 x double> [ %cond50.i177216, %cond.true58.i ], [ %cond50.i189, %cond.true52.i ], [ %cond50.i177, %cond.end55.i ]
  %cmp3.i51131466387117147172208264 = phi i1 [ %cmp3.i51131466387117147172221, %cond.true58.i ], [ true, %cond.true52.i ], [ false, %cond.end55.i ]
  %cmp9.i1725496090114150169211261 = phi i1 [ true, %cond.true58.i ], [ false, %cond.true52.i ], [ false, %cond.end55.i ]
  %cond56.i215257 = phi <3 x double> [ %cond56.i228, %cond.true58.i ], [ %hlsl.wave.prefix.product53.i, %cond.true52.i ], [ zeroinitializer, %cond.end55.i ]
  %hlsl.wave.prefix.product65.i = call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.spv.wave.prefix.product.v3f64(<3 x double> %.pre-phi454) [ "convergencectrl"(token %0) ]
  br i1 %cmp.i, label %cond.true70.i, label %cond.end73.i

cond.true70.i:                                    ; preds = %cond.true64.i
  %hlsl.wave.prefix.product71.i = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.spv.wave.prefix.product.v4f64(<4 x double> %4) [ "convergencectrl"(token %0) ]
  br i1 %cmp3.i51131466387117147172208264, label %cond.true76.i, label %cond.end79.i

cond.end73.i:                                     ; preds = %cond.true64.i
  br i1 %cmp3.i51131466387117147172208264, label %cond.true76.i, label %cond.end79.i

cond.true76.i:                                    ; preds = %cond.true70.i, %cond.end73.i
  %cond74.i346 = phi <4 x double> [ %hlsl.wave.prefix.product71.i, %cond.true70.i ], [ zeroinitializer, %cond.end73.i ]
  %hlsl.wave.prefix.product77.i = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.spv.wave.prefix.product.v4f64(<4 x double> %4) [ "convergencectrl"(token %0) ]
  br i1 %cmp9.i1725496090114150169211261, label %cond.true82.i, label %cond.true88.i

cond.end79.i:                                     ; preds = %cond.true70.i, %cond.end73.i
  %cond74.i331 = phi <4 x double> [ %hlsl.wave.prefix.product71.i, %cond.true70.i ], [ zeroinitializer, %cond.end73.i ]
  br i1 %cmp9.i1725496090114150169211261, label %cond.true82.i, label %cond.true88.i

cond.true82.i:                                    ; preds = %cond.true76.i, %cond.end79.i
  %cond80.i392 = phi <4 x double> [ %hlsl.wave.prefix.product77.i, %cond.true76.i ], [ zeroinitializer, %cond.end79.i ]
  %cond74.i331378 = phi <4 x double> [ %cond74.i346, %cond.true76.i ], [ %cond74.i331, %cond.end79.i ]
  %hlsl.wave.prefix.product83.i = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.spv.wave.prefix.product.v4f64(<4 x double> %4) [ "convergencectrl"(token %0) ]
  br label %cond.true88.i

cond.true88.i:                                    ; preds = %cond.end79.i, %cond.true76.i, %cond.true82.i
  %cond86.i438 = phi <4 x double> [ %hlsl.wave.prefix.product83.i, %cond.true82.i ], [ zeroinitializer, %cond.true76.i ], [ zeroinitializer, %cond.end79.i ]
  %cond74.i331363437 = phi <4 x double> [ %cond74.i331378, %cond.true82.i ], [ %cond74.i346, %cond.true76.i ], [ %cond74.i331, %cond.end79.i ]
  %cond80.i377424 = phi <4 x double> [ %cond80.i392, %cond.true82.i ], [ %hlsl.wave.prefix.product77.i, %cond.true76.i ], [ zeroinitializer, %cond.end79.i ]
  %hlsl.wave.prefix.product89.i = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.spv.wave.prefix.product.v4f64(<4 x double> %4) [ "convergencectrl"(token %0) ]
  br label %_Z4mainDv3_j.exit

_Z4mainDv3_j.exit:                                ; preds = %cond.end13.i, %cond.true88.i
  %cond86.i423 = phi <4 x double> [ %cond86.i438, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond74.i331363422 = phi <4 x double> [ %cond74.i331363437, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond62.i256286330364421 = phi <3 x double> [ %cond62.i270, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond50.i177203255287329365420 = phi <3 x double> [ %cond50.i177203269, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond38.i121143176204254288328366419 = phi <2 x double> [ %cond38.i131, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond26.i6684120144175205253289327367418 = phi <2 x double> [ %cond26.i6684130, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond14.i33456486118146173207251291325369417 = phi double [ %cond14.i34, %cond.true88.i ], [ 0.000000e+00, %cond.end13.i ]
  %cond.i31329476288116148171209249293324370416 = phi double [ %cond.i31330, %cond.true88.i ], [ 0.000000e+00, %cond.end13.i ]
  %cond8.i1527486189115149170210248294323371415 = phi double [ %cond8.i1528, %cond.true88.i ], [ 0.000000e+00, %cond.end13.i ]
  %cond20.i505991113151168212246296321372414 = phi double [ %hlsl.wave.prefix.product17.i, %cond.true88.i ], [ 0.000000e+00, %cond.end13.i ]
  %cond32.i92112152167213245297320373413 = phi <2 x double> [ %cond32.i92122, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond44.i153166214244298319374412 = phi <2 x double> [ %hlsl.wave.prefix.product41.i, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond56.i215243299318375411 = phi <3 x double> [ %cond56.i215257, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond68.i300317376410 = phi <3 x double> [ %hlsl.wave.prefix.product65.i, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond80.i377409 = phi <4 x double> [ %cond80.i377424, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %cond92.i = phi reassoc nnan ninf nsz arcp afn <4 x double> [ %hlsl.wave.prefix.product89.i, %cond.true88.i ], [ zeroinitializer, %cond.end13.i ]
  %8 = tail call target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f64_12_1t(i32 0, i32 5, i32 1, i32 0, ptr nonnull @.str.10)
  %9 = tail call target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f64_12_1t(i32 0, i32 4, i32 1, i32 0, ptr nonnull @.str.8)
  %10 = tail call target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f64_12_1t(i32 0, i32 3, i32 1, i32 0, ptr nonnull @.str.6)
  %11 = tail call target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f64_12_1t(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.4)
  %12 = tail call target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f64_12_1t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  call void @llvm.lifetime.start.p0(ptr nonnull %scalars.i) #5
  store double %cond.i31329476288116148171209249293324370416, ptr %scalars.i, align 8, !tbaa !7
  %arrayinit.element.i = getelementptr inbounds nuw i8, ptr %scalars.i, i64 8
  store double %cond8.i1527486189115149170210248294323371415, ptr %arrayinit.element.i, align 8, !tbaa !7
  %arrayinit.element93.i = getelementptr inbounds nuw i8, ptr %scalars.i, i64 16
  store double %cond14.i33456486118146173207251291325369417, ptr %arrayinit.element93.i, align 8, !tbaa !7
  %arrayinit.element94.i = getelementptr inbounds nuw i8, ptr %scalars.i, i64 24
  store double %cond20.i505991113151168212246296321372414, ptr %arrayinit.element94.i, align 8, !tbaa !7
  call void @llvm.lifetime.start.p0(ptr nonnull %vec2s.i) #5
  store <2 x double> %cond26.i6684120144175205253289327367418, ptr %vec2s.i, align 8, !tbaa !6
  %arrayinit.element97.i = getelementptr inbounds nuw i8, ptr %vec2s.i, i64 16
  store <2 x double> %cond32.i92112152167213245297320373413, ptr %arrayinit.element97.i, align 8, !tbaa !6
  %arrayinit.element102.i = getelementptr inbounds nuw i8, ptr %vec2s.i, i64 32
  store <2 x double> %cond38.i121143176204254288328366419, ptr %arrayinit.element102.i, align 8, !tbaa !6
  %arrayinit.element107.i = getelementptr inbounds nuw i8, ptr %vec2s.i, i64 48
  store <2 x double> %cond44.i153166214244298319374412, ptr %arrayinit.element107.i, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %vec3s.i) #5
  store <3 x double> %cond50.i177203255287329365420, ptr %vec3s.i, align 8, !tbaa !6
  %arrayinit.element118.i = getelementptr inbounds nuw i8, ptr %vec3s.i, i64 24
  store <3 x double> %cond56.i215243299318375411, ptr %arrayinit.element118.i, align 8, !tbaa !6
  %arrayinit.element125.i = getelementptr inbounds nuw i8, ptr %vec3s.i, i64 48
  store <3 x double> %cond62.i256286330364421, ptr %arrayinit.element125.i, align 8, !tbaa !6
  %arrayinit.element132.i = getelementptr inbounds nuw i8, ptr %vec3s.i, i64 72
  store <3 x double> %cond68.i300317376410, ptr %arrayinit.element132.i, align 8, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %vec4s.i) #5
  store <4 x double> %cond74.i331363422, ptr %vec4s.i, align 8, !tbaa !6
  %arrayinit.element147.i = getelementptr inbounds nuw i8, ptr %vec4s.i, i64 32
  store <4 x double> %cond80.i377409, ptr %arrayinit.element147.i, align 8, !tbaa !6
  %arrayinit.element156.i = getelementptr inbounds nuw i8, ptr %vec4s.i, i64 64
  store <4 x double> %cond86.i423, ptr %arrayinit.element156.i, align 8, !tbaa !6
  %arrayinit.element165.i = getelementptr inbounds nuw i8, ptr %vec4s.i, i64 96
  store <4 x double> %cond92.i, ptr %arrayinit.element165.i, align 8, !tbaa !6
  %idxprom.i = zext i32 %2 to i64
  %arrayidx.i = getelementptr inbounds nuw [8 x i8], ptr %scalars.i, i64 %idxprom.i
  %13 = load double, ptr %arrayidx.i, align 8, !tbaa !7
  %14 = tail call noundef align 8 dereferenceable(32) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f64_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) %12, i32 %2)
  store double %13, ptr addrspace(11) %14, align 8
  %arrayidx176.i = getelementptr inbounds nuw [16 x i8], ptr %vec2s.i, i64 %idxprom.i
  %15 = load <2 x double>, ptr %arrayidx176.i, align 8, !tbaa !6
  %16 = tail call noundef align 8 dereferenceable(32) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f64_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) %11, i32 %2)
  %17 = extractelement <2 x double> %15, i64 0
  store double %17, ptr addrspace(11) %16, align 8
  %18 = extractelement <2 x double> %15, i64 1
  %19 = getelementptr inbounds nuw i8, ptr addrspace(11) %16, i64 8
  store double %18, ptr addrspace(11) %19, align 8
  %arrayidx179.i = getelementptr inbounds nuw [24 x i8], ptr %vec3s.i, i64 %idxprom.i
  %20 = load <3 x double>, ptr %arrayidx179.i, align 8, !tbaa !6
  %21 = tail call noundef align 8 dereferenceable(32) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f64_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) %10, i32 %2)
  %22 = extractelement <3 x double> %20, i64 0
  store double %22, ptr addrspace(11) %21, align 8
  %23 = extractelement <3 x double> %20, i64 1
  %24 = getelementptr inbounds nuw i8, ptr addrspace(11) %21, i64 8
  store double %23, ptr addrspace(11) %24, align 8
  %25 = extractelement <3 x double> %20, i64 2
  %26 = getelementptr inbounds nuw i8, ptr addrspace(11) %21, i64 16
  store double %25, ptr addrspace(11) %26, align 8
  %arrayidx182.i = getelementptr inbounds nuw [32 x i8], ptr %vec4s.i, i64 %idxprom.i
  %27 = load <4 x double>, ptr %arrayidx182.i, align 8, !tbaa !6
  %28 = tail call noundef align 8 dereferenceable(32) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f64_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) %9, i32 %2)
  store <4 x double> %27, ptr addrspace(11) %28, align 8, !tbaa !6
  %hlsl.wave.prefix.product184.i = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.spv.wave.prefix.product.v4f64(<4 x double> <double 2.000000e+00, double 3.000000e+00, double 5.000000e+00, double 7.000000e+00>) [ "convergencectrl"(token %0) ]
  %29 = tail call noundef align 8 dereferenceable(32) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f64_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) %8, i32 %2)
  store <4 x double> %hlsl.wave.prefix.product184.i, ptr addrspace(11) %29, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %vec4s.i) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %vec3s.i) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %vec2s.i) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %scalars.i) #5
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.spv.thread.id.in.group.i32(i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare double @llvm.spv.wave.prefix.product.f64(double) #0

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <2 x double> @llvm.spv.wave.prefix.product.v2f64(<2 x double>) #0

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <3 x double> @llvm.spv.wave.prefix.product.v3f64(<3 x double>) #0

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x double> @llvm.spv.wave.prefix.product.v4f64(<4 x double>) #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f64_12_0t(i32, i32, i32, i32, ptr) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f64_12_1t(i32, i32, i32, i32, ptr) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f64_12_0t.i32(target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 0), i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f64_12_1t.i32(target("spirv.VulkanBuffer", [0 x <4 x double>], 12, 1), i32) #4

attributes #0 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #1 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none, target_mem: none) "frame-pointer"="all" "hlsl.numthreads"="4,1,1" "hlsl.shader"="compute" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}
!llvm.errno.tbaa = !{!2}

!0 = !{i32 7, !"frame-pointer", i32 2}
!1 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git dfab397600fb548e895d2c28832d5eaeea3f8bb1)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = !{!4, !4, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"double", !4, i64 0}
