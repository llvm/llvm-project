; Source:
;
; #pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable
; void  test(__read_only image2d_t src,
;            __read_only image2d_t ref,
;            sampler_t sampler) {
;
;   intel_sub_group_avc_ime_payload_t ime_payload;
;   ime_payload = intel_sub_group_avc_ime_set_inter_base_multi_reference_penalty(
;     0, ime_payload);
;
;   intel_sub_group_avc_ime_result_t ime_result;
;   intel_sub_group_avc_ime_get_motion_vectors(ime_result);
;
;   intel_sub_group_avc_ref_payload_t ref_payload;
;   ref_payload = intel_sub_group_avc_ref_set_inter_shape_penalty(0, ref_payload);
;
;   intel_sub_group_avc_ref_result_t ref_result;
;   intel_sub_group_avc_ref_get_inter_distortions(ref_result);
;
;   intel_sub_group_avc_sic_payload_t sic_payload;
;   sic_payload = intel_sub_group_avc_sic_set_motion_vector_cost_function(
;     0, 0, 0, sic_payload);
;
;   intel_sub_group_avc_sic_result_t sic_result;
;   intel_sub_group_avc_sic_get_inter_distortions(sic_result);
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s

; The test checks that 'cl_intel_device_side_avc_motion_estimation' wrapper built-ins correctly
; translated to 'SPV_INTEL_device_side_avc_motion_estimation' extension instructions.

; CHECK: Capability Groups
; CHECK: Capability SubgroupAvcMotionEstimationINTEL

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK: TypeAvcImePayloadINTEL [[ImePayloadTy:[0-9]+]]
; CHECK: TypeAvcImeResultINTEL  [[ImeResultTy:[0-9]+]]
; CHECK: TypeAvcRefPayloadINTEL [[RefPayloadTy:[0-9]+]]
; CHECK: TypeAvcRefResultINTEL  [[RefResultTy:[0-9]+]]
; CHECK: TypeAvcSicPayloadINTEL [[SicPayloadTy:[0-9]+]]
; CHECK: TypeAvcSicResultINTEL  [[SicResultTy:[0-9]+]]
; CHECK: TypeAvcMcePayloadINTEL [[McePayloadTy:[0-9]+]]
; CHECK: TypeAvcMceResultINTEL  [[MceResultTy:[0-9]+]]

%opencl.intel_sub_group_avc_ime_payload_t = type opaque
%opencl.intel_sub_group_avc_ime_result_t = type opaque
%opencl.intel_sub_group_avc_ref_payload_t = type opaque
%opencl.intel_sub_group_avc_ref_result_t = type opaque
%opencl.intel_sub_group_avc_sic_payload_t = type opaque
%opencl.intel_sub_group_avc_sic_result_t = type opaque

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @test() #0 {
entry:

  %ime_payload = alloca %opencl.intel_sub_group_avc_ime_payload_t*, align 8
  %ime_result = alloca %opencl.intel_sub_group_avc_ime_result_t*, align 8
  %ref_payload = alloca %opencl.intel_sub_group_avc_ref_payload_t*, align 8
  %ref_result = alloca %opencl.intel_sub_group_avc_ref_result_t*, align 8
  %sic_payload = alloca %opencl.intel_sub_group_avc_sic_payload_t*, align 8
  %sic_result = alloca %opencl.intel_sub_group_avc_sic_result_t*, align 8

; CHECK:  Load [[ImePayloadTy]] [[ImePayload:[0-9]+]]
; CHECK:  Load [[ImeResultTy]]  [[ImeResult:[0-9]+]]
; CHECK:  Load [[RefPayloadTy]] [[RefPayload:[0-9]+]]
; CHECK:  Load [[RefResultTy]]  [[RefResult:[0-9]+]]
; CHECK:  Load [[SicPayloadTy]] [[SicPayload:[0-9]+]]
; CHECK:  Load [[SicResultTy]]  [[SicResult:[0-9]+]]

  %0 = load %opencl.intel_sub_group_avc_ime_payload_t*, %opencl.intel_sub_group_avc_ime_payload_t** %ime_payload, align 8
  %1 = load %opencl.intel_sub_group_avc_ime_result_t*, %opencl.intel_sub_group_avc_ime_result_t** %ime_result, align 8
  %2 = load %opencl.intel_sub_group_avc_ref_payload_t*, %opencl.intel_sub_group_avc_ref_payload_t** %ref_payload, align 8
  %3 = load %opencl.intel_sub_group_avc_ref_result_t*, %opencl.intel_sub_group_avc_ref_result_t** %ref_result, align 8
  %4 = load %opencl.intel_sub_group_avc_sic_payload_t*, %opencl.intel_sub_group_avc_sic_payload_t** %sic_payload, align 8
  %5 = load %opencl.intel_sub_group_avc_sic_result_t*, %opencl.intel_sub_group_avc_sic_result_t** %sic_result, align 8

; CHECK:      SubgroupAvcImeConvertToMcePayloadINTEL
; CHECK-SAME:     [[McePayloadTy]] [[ImeMcePayloadConv:[0-9]+]] [[ImePayload]]
; CHECK:      SubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL
; CHECK-SAME:     [[McePayloadTy]] [[McePayloadRet0:[0-9]+]] {{.*}} [[ImeMcePayloadConv]]
; CHECK:      SubgroupAvcMceConvertToImePayloadINTEL
; CHECK-SAME:     [[ImePayloadTy]] {{.*}} [[McePayloadRet0]]
  %call0 = call spir_func %opencl.intel_sub_group_avc_ime_payload_t* @_Z62intel_sub_group_avc_ime_set_inter_base_multi_reference_penaltyh37ocl_intel_sub_group_avc_ime_payload_t(i8 zeroext 0, %opencl.intel_sub_group_avc_ime_payload_t* %0) #2

; CHECK:      SubgroupAvcImeConvertToMceResultINTEL
; CHECK-SAME:     [[MceResultTy]] [[ImeMceResultConv:[0-9]+]] [[ImeResult]]
; CHECK:      SubgroupAvcMceGetMotionVectorsINTEL {{.*}} [[ImeMceResultConv]]
  %call1 = call spir_func i64 @_Z42intel_sub_group_avc_ime_get_motion_vectors36ocl_intel_sub_group_avc_ime_result_t(%opencl.intel_sub_group_avc_ime_result_t* %1) #2

; CHECK:      SubgroupAvcRefConvertToMcePayloadINTEL
; CHECK-SAME:     [[McePayloadTy]] [[RefMcePayloadConv:[0-9]+]] [[RefPayload]]
; CHECK:      SubgroupAvcMceSetInterShapePenaltyINTEL
; CHECK-SAME:     [[McePayloadTy]] [[McePayloadRet1:[0-9]+]] {{.*}} [[RefMcePayloadConv]]
; CHECK:      SubgroupAvcMceConvertToRefPayloadINTEL
; CHECK-SAME:     [[RefPayloadTy]] {{.*}} [[McePayloadRet1]]
  %call2 = call spir_func %opencl.intel_sub_group_avc_ref_payload_t* @_Z47intel_sub_group_avc_ref_set_inter_shape_penaltym37ocl_intel_sub_group_avc_ref_payload_t(i64 0, %opencl.intel_sub_group_avc_ref_payload_t* %2) #2

; CHECK:      SubgroupAvcRefConvertToMceResultINTEL
; CHECK-SAME:     [[MceResultTy]] [[RefMceResultConv:[0-9]+]] [[RefResult]]
; CHECK:      SubgroupAvcMceGetInterDistortionsINTEL {{.*}} [[RefMceResultConv]]
  %call3 = call spir_func zeroext i16 @_Z45intel_sub_group_avc_ref_get_inter_distortions36ocl_intel_sub_group_avc_ref_result_t(%opencl.intel_sub_group_avc_ref_result_t* %3) #2

; CHECK:      SubgroupAvcSicConvertToMcePayloadINTEL
; CHECK-SAME:     [[McePayloadTy]] [[SicMcePayloadConv:[0-9]+]] [[SicPayload]]
; CHECK:      SubgroupAvcMceSetMotionVectorCostFunctionINTEL
; CHECK-SAME:     [[McePayloadTy]] [[McePayloadRet2:[0-9]+]] {{.*}} [[SicMcePayloadConv]]
; CHECK:      SubgroupAvcMceConvertToSicPayloadINTEL
; CHECK-SAME:     [[SicPayloadTy]] {{.*}} [[McePayloadRet2]]
  %call4 = call spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z55intel_sub_group_avc_sic_set_motion_vector_cost_functionmDv2_jh37ocl_intel_sub_group_avc_sic_payload_t(i64 0, <2 x i32> zeroinitializer, i8 zeroext 0, %opencl.intel_sub_group_avc_sic_payload_t* %4) #2

; CHECK:      SubgroupAvcSicConvertToMceResultINTEL
; CHECK-SAME:     [[MceResultTy]] [[SicMceResultConv:[0-9]+]] [[SicResult]]
; CHECK:      SubgroupAvcMceGetInterDistortionsINTEL {{.*}} [[SicMceResultConv]]
  %call5 = call spir_func zeroext i16 @_Z45intel_sub_group_avc_sic_get_inter_distortions36ocl_intel_sub_group_avc_sic_result_t(%opencl.intel_sub_group_avc_sic_result_t* %5) #2
  ret void
}

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_payload_t* @_Z62intel_sub_group_avc_ime_set_inter_base_multi_reference_penaltyh37ocl_intel_sub_group_avc_ime_payload_t(i8 zeroext, %opencl.intel_sub_group_avc_ime_payload_t*) #1

; Function Attrs: convergent
declare spir_func i64 @_Z42intel_sub_group_avc_ime_get_motion_vectors36ocl_intel_sub_group_avc_ime_result_t(%opencl.intel_sub_group_avc_ime_result_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ref_payload_t* @_Z47intel_sub_group_avc_ref_set_inter_shape_penaltym37ocl_intel_sub_group_avc_ref_payload_t(i64, %opencl.intel_sub_group_avc_ref_payload_t*) #1

; Function Attrs: convergent
declare spir_func zeroext i16 @_Z45intel_sub_group_avc_ref_get_inter_distortions36ocl_intel_sub_group_avc_ref_result_t(%opencl.intel_sub_group_avc_ref_result_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z55intel_sub_group_avc_sic_set_motion_vector_cost_functionmDv2_jh37ocl_intel_sub_group_avc_sic_payload_t(i64, <2 x i32>, i8 zeroext, %opencl.intel_sub_group_avc_sic_payload_t*) #1

; Function Attrs: convergent
declare spir_func zeroext i16 @_Z45intel_sub_group_avc_sic_get_inter_distortions36ocl_intel_sub_group_avc_sic_result_t(%opencl.intel_sub_group_avc_sic_result_t*) #1

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{}
!3 = !{!"cl_images"}
!4 = !{!"clang version 6.0.0"}
