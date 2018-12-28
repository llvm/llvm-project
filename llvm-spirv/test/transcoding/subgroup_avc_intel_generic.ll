; Source:
; #pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable
; void foo() {
;   intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penalty(0, 0);
;   intel_sub_group_avc_mce_get_default_inter_shape_penalty(0, 0);
;   intel_sub_group_avc_mce_get_default_intra_luma_shape_penalty(0, 0);
;   intel_sub_group_avc_mce_get_default_inter_motion_vector_cost_table(0, 0);
;
;   intel_sub_group_avc_ime_payload_t ime_payload;
;   intel_sub_group_avc_ime_initialize(0, 0, 0);
;   intel_sub_group_avc_ime_set_single_reference(0, 0, ime_payload);
;   intel_sub_group_avc_ime_ref_window_size(0, 0);
;   intel_sub_group_ime_ref_window_size(0, 0); // This function defined in the spec
;   intel_sub_group_avc_ime_adjust_ref_offset(0, 0, 0, 0);
;   intel_sub_group_avc_ime_set_max_motion_vector_count(0, ime_payload);
;
;   intel_sub_group_avc_ime_result_single_reference_streamout_t sstreamout;
;   intel_sub_group_avc_ime_get_single_reference_streamin(sstreamout);
;
;   intel_sub_group_avc_ime_result_dual_reference_streamout_t dstreamout;
;   intel_sub_group_avc_ime_get_dual_reference_streamin(dstreamout);
;
;   intel_sub_group_avc_ime_result_t ime_result;
;   intel_sub_group_avc_ime_get_border_reached(0i, ime_result);
;
;   intel_sub_group_avc_fme_initialize(0, 0, 0, 0, 0, 0, 0);
;   intel_sub_group_avc_bme_initialize(0, 0, 0, 0, 0, 0, 0, 0);
;
;   intel_sub_group_avc_ref_payload_t ref_payload;
;   intel_sub_group_avc_ref_set_bidirectional_mix_disable(ref_payload);
;
;   intel_sub_group_avc_sic_initialize(0);
;   intel_sub_group_avc_sic_payload_t sic_payload;
;   intel_sub_group_avc_sic_configure_ipe(0, 0, 0, 0, 0, 0, 0, sic_payload);
;   intel_sub_group_avc_sic_configure_ipe(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sic_payload);
;
;   intel_sub_group_avc_sic_result_t sic_result;
;   intel_sub_group_avc_sic_get_best_ipe_luma_distortion(sic_result);
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s

; The test checks several (not all) 'cl_intel_device_side_avc_motion_estimation'
; extension built-ins.
; Checks that both spelling for 'intel_sub_group_avc_ime_ref_window_size' are
; accepted by the SPIRVWriter:
; 'intel_sub_group_avc_ime_ref_window_size()' (correct name)
; 'intel_sub_group_ime_ref_window_size()' (name defined in the spec).

; CHECK: Capability Groups
; CHECK: Capability SubgroupAvcMotionEstimationINTEL
; CHECK: Capability SubgroupAvcMotionEstimationIntraINTEL
; CHECK: Capability SubgroupAvcMotionEstimationChromaINTEL

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK: TypeAvcImePayloadINTEL                        [[ImePayloadTy:[0-9]+]]
; CHECK: TypeAvcImeSingleReferenceStreaminINTEL        [[ImeSRefInTy:[0-9]+]]
; CHECK: TypeAvcImeDualReferenceStreaminINTEL          [[ImeDRefInTy:[0-9]+]]
; CHECK: TypeAvcImeResultSingleReferenceStreamoutINTEL [[ImeSRefOutTy:[0-9]+]]
; CHECK: TypeAvcImeResultDualReferenceStreamoutINTEL   [[ImeDRefOutTy:[0-9]+]]
; CHECK: TypeAvcImeResultINTEL                         [[ImeResultTy:[0-9]+]]
; CHECK: TypeAvcRefPayloadINTEL                        [[RefPayloadTy:[0-9]+]]
; CHECK: TypeAvcSicPayloadINTEL                        [[SicPayloadTy:[0-9]+]]
; CHECK: TypeAvcSicResultINTEL                         [[SicResultTy:[0-9]+]]

%opencl.intel_sub_group_avc_ime_payload_t = type opaque
%opencl.intel_sub_group_avc_ime_single_reference_streamin_t = type opaque
%opencl.intel_sub_group_avc_ime_dual_reference_streamin_t = type opaque
%opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t = type opaque
%opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t = type opaque
%opencl.intel_sub_group_avc_ime_result_t = type opaque
%opencl.intel_sub_group_avc_ref_payload_t = type opaque
%opencl.intel_sub_group_avc_sic_payload_t = type opaque
%opencl.intel_sub_group_avc_sic_result_t = type opaque
%opencl.intel_sub_group_avc_ref_result_t = type opaque

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @foo() #0 {
entry:
  %ime_payload = alloca %opencl.intel_sub_group_avc_ime_payload_t*, align 8
  %sstreamin = alloca %opencl.intel_sub_group_avc_ime_single_reference_streamin_t*, align 8
  %dstreamin = alloca %opencl.intel_sub_group_avc_ime_dual_reference_streamin_t*, align 8
  %sstreamout = alloca %opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t*, align 8
  %dstreamout = alloca %opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t*, align 8
  %ime_result = alloca %opencl.intel_sub_group_avc_ime_result_t*, align 8
  %ref_payload = alloca %opencl.intel_sub_group_avc_ref_payload_t*, align 8
  %sic_payload = alloca %opencl.intel_sub_group_avc_sic_payload_t*, align 8
  %sic_result = alloca %opencl.intel_sub_group_avc_sic_result_t*, align 8

; CHECK:  Load [[ImePayloadTy]] [[ImePayload:[0-9]+]]
; CHECK:  Load [[ImeSRefOutTy]] [[ImeSRefOut:[0-9]+]]
; CHECK:  Load [[ImeDRefOutTy]] [[ImeDRefOut:[0-9]+]]
; CHECK:  Load [[ImeResultTy]]  [[ImeResult:[0-9]+]]
; CHECK:  Load [[RefPayloadTy]] [[RefPayload:[0-9]+]]
; CHECK:  Load [[SicPayloadTy]] [[SicPayload:[0-9]+]]
; CHECK:  Load [[SicResultTy]]  [[SicResult:[0-9]+]]

  %0 = load %opencl.intel_sub_group_avc_ime_payload_t*, %opencl.intel_sub_group_avc_ime_payload_t** %ime_payload, align 8
  %1 = load %opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t*, %opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t** %sstreamout, align 8
  %2 = load %opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t*, %opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t** %dstreamout, align 8
  %3 = load %opencl.intel_sub_group_avc_ime_result_t*, %opencl.intel_sub_group_avc_ime_result_t** %ime_result, align 8
  %4 = load %opencl.intel_sub_group_avc_ref_payload_t*, %opencl.intel_sub_group_avc_ref_payload_t** %ref_payload, align 8
  %5 = load %opencl.intel_sub_group_avc_sic_payload_t*, %opencl.intel_sub_group_avc_sic_payload_t** %sic_payload, align 8
  %6 = load %opencl.intel_sub_group_avc_sic_result_t*, %opencl.intel_sub_group_avc_sic_result_t** %sic_result, align 8

; CHECK:  SubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL
; CHECK:  SubgroupAvcMceGetDefaultInterShapePenaltyINTEL
; CHECK:  SubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL
; CHECK:  SubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL

  %call = call spir_func zeroext i8 @_Z70intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penaltyhh(i8 zeroext 0, i8 zeroext 0) #2
  %call1 = call spir_func i64 @_Z55intel_sub_group_avc_mce_get_default_inter_shape_penaltyhh(i8 zeroext 0, i8 zeroext 0) #2
  %call2 = call spir_func i32 @_Z60intel_sub_group_avc_mce_get_default_intra_luma_shape_penaltyhh(i8 zeroext 0, i8 zeroext 0) #2
  %call3 = call spir_func <2 x i32> @_Z66intel_sub_group_avc_mce_get_default_inter_motion_vector_cost_tablehh(i8 zeroext 0, i8 zeroext 0) #2

; CHECK:  SubgroupAvcImeInitializeINTEL [[ImePayloadTy]]
  %call4 = call spir_func %opencl.intel_sub_group_avc_ime_payload_t* @_Z34intel_sub_group_avc_ime_initializeDv2_thh(<2 x i16> zeroinitializer, i8 zeroext 0, i8 zeroext 0) #2

; CHECK:  SubgroupAvcImeSetSingleReferenceINTEL [[ImePayloadTy]] {{.*}} [[ImePayload]]
  %call5 = call spir_func %opencl.intel_sub_group_avc_ime_payload_t* @_Z44intel_sub_group_avc_ime_set_single_referenceDv2_sh37ocl_intel_sub_group_avc_ime_payload_t(<2 x i16> zeroinitializer, i8 zeroext 0, %opencl.intel_sub_group_avc_ime_payload_t* %0) #2

; CHECK:  SubgroupAvcImeRefWindowSizeINTEL
; CHECK:  SubgroupAvcImeRefWindowSizeINTEL
; CHECK:  SubgroupAvcImeAdjustRefOffsetINTEL
  %call6 = call spir_func <2 x i16> @_Z39intel_sub_group_avc_ime_ref_window_sizehc(i8 zeroext 0, i8 signext 0) #2
  %call6i = call spir_func <2 x i16> @_Z35intel_sub_group_ime_ref_window_sizehc(i8 zeroext 0, i8 signext 0) #2
  %call7 = call spir_func <2 x i16> @_Z41intel_sub_group_avc_ime_adjust_ref_offsetDv2_sDv2_tS0_S0_(<2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer) #2

; CHECK:  SubgroupAvcImeSetMaxMotionVectorCountINTEL [[ImePayloadTy]] {{.*}} [[ImePayload]]
  %call8 = call spir_func %opencl.intel_sub_group_avc_ime_payload_t* @_Z51intel_sub_group_avc_ime_set_max_motion_vector_counth37ocl_intel_sub_group_avc_ime_payload_t(i8 zeroext 0, %opencl.intel_sub_group_avc_ime_payload_t* %0) #2

; CHECK:  SubgroupAvcImeGetSingleReferenceStreaminINTEL [[ImeSRefInTy]] {{.*}} [[ImeSRefOut]]
; CHECK:  SubgroupAvcImeGetDualReferenceStreaminINTEL [[ImeDRefInTy]] {{.*}} [[ImeDRefOut]]
  %call9 = call spir_func %opencl.intel_sub_group_avc_ime_single_reference_streamin_t* @_Z53intel_sub_group_avc_ime_get_single_reference_streamin63ocl_intel_sub_group_avc_ime_result_single_reference_streamout_t(%opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t* %1) #2
  %call10 = call spir_func %opencl.intel_sub_group_avc_ime_dual_reference_streamin_t* @_Z51intel_sub_group_avc_ime_get_dual_reference_streamin61ocl_intel_sub_group_avc_ime_result_dual_reference_streamout_t(%opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t* %2) #2

; CHECK:  SubgroupAvcImeGetBorderReachedINTEL {{.*}} [[ImeResult]]
  %call11 = call spir_func zeroext i8 @_Z42intel_sub_group_avc_ime_get_border_reachedh36ocl_intel_sub_group_avc_ime_result_t(i8 zeroext 0, %opencl.intel_sub_group_avc_ime_result_t* %3) #2

; CHECK:  SubgroupAvcFmeInitializeINTEL [[RefPayloadTy]]
; CHECK:  SubgroupAvcBmeInitializeINTEL [[RefPayloadTy]]
  %call12 = call spir_func %opencl.intel_sub_group_avc_ref_payload_t* @_Z34intel_sub_group_avc_fme_initializeDv2_tmhhhhh(<2 x i16> zeroinitializer, i64 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0) #2
  %call13 = call spir_func %opencl.intel_sub_group_avc_ref_payload_t* @_Z34intel_sub_group_avc_bme_initializeDv2_tmhhhhhh(<2 x i16> zeroinitializer, i64 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0) #2

; CHECK:  SubgroupAvcRefSetBidirectionalMixDisableINTEL [[RefPayloadTy]] {{.*}} [[RefPayload]]
  %call14 = call spir_func %opencl.intel_sub_group_avc_ref_payload_t* @_Z53intel_sub_group_avc_ref_set_bidirectional_mix_disable37ocl_intel_sub_group_avc_ref_payload_t(%opencl.intel_sub_group_avc_ref_payload_t* %4) #2

; CHECK:  SubgroupAvcSicInitializeINTEL [[SicPayloadTy]]
  %call15 = call spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z34intel_sub_group_avc_sic_initializeDv2_t(<2 x i16> zeroinitializer) #2

; CHECK:  SubgroupAvcSicConfigureIpeLumaINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
; CHECK:  SubgroupAvcSicConfigureIpeLumaChromaINTEL [[SicPayloadTy]] {{.*}} [[SicPayload]]
  %call16 = call spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z37intel_sub_group_avc_sic_configure_ipehhhhhhh37ocl_intel_sub_group_avc_sic_payload_t(i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, %opencl.intel_sub_group_avc_sic_payload_t* %5) #2
  %call17 = call spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z37intel_sub_group_avc_sic_configure_ipehhhhhhttth37ocl_intel_sub_group_avc_sic_payload_t(i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i8 zeroext 0, i16 zeroext 0, i16 zeroext 0, i16 zeroext 0, i8 zeroext 0, %opencl.intel_sub_group_avc_sic_payload_t* %5) #2

; CHECK:  SubgroupAvcSicGetBestIpeLumaDistortionINTEL {{.*}} [[SicResult]]
  %call18 = call spir_func zeroext i16 @_Z52intel_sub_group_avc_sic_get_best_ipe_luma_distortion36ocl_intel_sub_group_avc_sic_result_t(%opencl.intel_sub_group_avc_sic_result_t* %6) #2
  ret void
}

; Function Attrs: convergent
declare spir_func zeroext i8 @_Z70intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penaltyhh(i8 zeroext, i8 zeroext) #1

; Function Attrs: convergent
declare spir_func i64 @_Z55intel_sub_group_avc_mce_get_default_inter_shape_penaltyhh(i8 zeroext, i8 zeroext) #1

; Function Attrs: convergent
declare spir_func i32 @_Z60intel_sub_group_avc_mce_get_default_intra_luma_shape_penaltyhh(i8 zeroext, i8 zeroext) #1

; Function Attrs: convergent
declare spir_func <2 x i32> @_Z66intel_sub_group_avc_mce_get_default_inter_motion_vector_cost_tablehh(i8 zeroext, i8 zeroext) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_payload_t* @_Z34intel_sub_group_avc_ime_initializeDv2_thh(<2 x i16>, i8 zeroext, i8 zeroext) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_payload_t* @_Z44intel_sub_group_avc_ime_set_single_referenceDv2_sh37ocl_intel_sub_group_avc_ime_payload_t(<2 x i16>, i8 zeroext, %opencl.intel_sub_group_avc_ime_payload_t*) #1

; Function Attrs: convergent
declare spir_func <2 x i16> @_Z39intel_sub_group_avc_ime_ref_window_sizehc(i8 zeroext, i8 signext) #1

; Function Attrs: convergent
declare spir_func <2 x i16> @_Z35intel_sub_group_ime_ref_window_sizehc(i8 zeroext, i8 signext) #1

; Function Attrs: convergent
declare spir_func <2 x i16> @_Z41intel_sub_group_avc_ime_adjust_ref_offsetDv2_sDv2_tS0_S0_(<2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_payload_t* @_Z51intel_sub_group_avc_ime_set_max_motion_vector_counth37ocl_intel_sub_group_avc_ime_payload_t(i8 zeroext, %opencl.intel_sub_group_avc_ime_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_single_reference_streamin_t* @_Z53intel_sub_group_avc_ime_get_single_reference_streamin63ocl_intel_sub_group_avc_ime_result_single_reference_streamout_t(%opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_dual_reference_streamin_t* @_Z51intel_sub_group_avc_ime_get_dual_reference_streamin61ocl_intel_sub_group_avc_ime_result_dual_reference_streamout_t(%opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t*) #1

; Function Attrs: convergent
declare spir_func zeroext i8 @_Z42intel_sub_group_avc_ime_get_border_reachedh36ocl_intel_sub_group_avc_ime_result_t(i8 zeroext, %opencl.intel_sub_group_avc_ime_result_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ref_payload_t* @_Z34intel_sub_group_avc_fme_initializeDv2_tmhhhhh(<2 x i16>, i64, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ref_payload_t* @_Z34intel_sub_group_avc_bme_initializeDv2_tmhhhhhh(<2 x i16>, i64, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ref_payload_t* @_Z53intel_sub_group_avc_ref_set_bidirectional_mix_disable37ocl_intel_sub_group_avc_ref_payload_t(%opencl.intel_sub_group_avc_ref_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z34intel_sub_group_avc_sic_initializeDv2_t(<2 x i16>) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z37intel_sub_group_avc_sic_configure_ipehhhhhhh37ocl_intel_sub_group_avc_sic_payload_t(i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, %opencl.intel_sub_group_avc_sic_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_sic_payload_t* @_Z37intel_sub_group_avc_sic_configure_ipehhhhhhttth37ocl_intel_sub_group_avc_sic_payload_t(i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i16 zeroext, i16 zeroext, i16 zeroext, i8 zeroext, %opencl.intel_sub_group_avc_sic_payload_t*) #1

; Function Attrs: convergent
declare spir_func zeroext i16 @_Z52intel_sub_group_avc_sic_get_best_ipe_luma_distortion36ocl_intel_sub_group_avc_sic_result_t(%opencl.intel_sub_group_avc_sic_result_t*) #1

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
