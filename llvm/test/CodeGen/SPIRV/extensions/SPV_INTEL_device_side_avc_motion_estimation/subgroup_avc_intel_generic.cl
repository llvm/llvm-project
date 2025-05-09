// RUN: clang -cc1 -O1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-llvm %s -o %t.ll
// RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation %t.ll -o - | FileCheck %s
void foo(intel_sub_group_avc_ime_payload_t ime_payload,
    intel_sub_group_avc_ime_result_single_reference_streamout_t sstreamout,
         intel_sub_group_avc_ime_result_dual_reference_streamout_t dstreamout,
         intel_sub_group_avc_ime_result_t ime_result,
         intel_sub_group_avc_mce_result_t mce_result,
         intel_sub_group_avc_ref_payload_t ref_payload,
         intel_sub_group_avc_sic_payload_t sic_payload,
         intel_sub_group_avc_sic_result_t sic_result,
         intel_sub_group_avc_mce_payload_t mce_payload) {
  intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penalty(0, 0);
  intel_sub_group_avc_mce_get_default_inter_shape_penalty(0, 0);
  intel_sub_group_avc_mce_get_default_intra_luma_shape_penalty(0, 0);
  intel_sub_group_avc_mce_get_default_inter_motion_vector_cost_table(0, 0);
  intel_sub_group_avc_mce_get_default_inter_direction_penalty(0, 0);
  intel_sub_group_avc_mce_get_default_intra_luma_mode_penalty(0, 0);

  intel_sub_group_avc_ime_initialize(0, 0, 0);
  intel_sub_group_avc_ime_set_single_reference(0, 0, ime_payload);
  intel_sub_group_avc_ime_set_dual_reference(0, 0, 0, ime_payload);
  intel_sub_group_avc_ime_ref_window_size(0, 0);
  intel_sub_group_avc_ime_ref_window_size(0, 0);
  intel_sub_group_avc_ime_adjust_ref_offset(0, 0, 0, 0);
  intel_sub_group_avc_ime_set_max_motion_vector_count(0, ime_payload);

  intel_sub_group_avc_ime_get_single_reference_streamin(sstreamout);

  intel_sub_group_avc_ime_get_dual_reference_streamin(dstreamout);

  intel_sub_group_avc_ime_get_border_reached(0i, ime_result);

  intel_sub_group_avc_ime_get_streamout_major_shape_distortions(sstreamout, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_distortions(dstreamout, 0, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_motion_vectors(sstreamout, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_motion_vectors(dstreamout, 0, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_reference_ids(sstreamout, 0);
  intel_sub_group_avc_ime_get_streamout_major_shape_reference_ids(dstreamout, 0, 0);

  intel_sub_group_avc_ime_set_dual_reference(0, 0, 0, ime_payload);
  intel_sub_group_avc_ime_set_weighted_sad(0, ime_payload);

  intel_sub_group_avc_ime_set_early_search_termination_threshold(0, ime_payload);

  intel_sub_group_avc_fme_initialize(0, 0, 0, 0, 0, 0, 0);
  intel_sub_group_avc_bme_initialize(0, 0, 0, 0, 0, 0, 0, 0);

  intel_sub_group_avc_ref_set_bidirectional_mix_disable(ref_payload);

  intel_sub_group_avc_sic_initialize(0);
  intel_sub_group_avc_sic_configure_ipe(0, 0, 0, 0, 0, 0, 0, sic_payload);
  intel_sub_group_avc_sic_configure_ipe(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sic_payload);

  intel_sub_group_avc_sic_configure_skc(0, 0, 0, 0, 0, sic_payload);

  intel_sub_group_avc_sic_set_skc_forward_transform_enable(0, sic_payload);
  intel_sub_group_avc_sic_set_block_based_raw_skip_sad(0, sic_payload);
  intel_sub_group_avc_sic_set_intra_luma_shape_penalty(0, sic_payload);
  intel_sub_group_avc_sic_set_intra_luma_mode_cost_function(0, 0, 0,
                                                            sic_payload);
  intel_sub_group_avc_sic_set_intra_chroma_mode_cost_function(0, sic_payload);

  intel_sub_group_avc_sic_get_best_ipe_luma_distortion(sic_result);
  intel_sub_group_avc_sic_get_motion_vector_mask(0, 0);

  intel_sub_group_avc_mce_set_source_interlaced_field_polarity(0, mce_payload);
  intel_sub_group_avc_mce_set_single_reference_interlaced_field_polarity(
      0, mce_payload);
  intel_sub_group_avc_mce_set_dual_reference_interlaced_field_polarities(
      0, 0, mce_payload);
  intel_sub_group_avc_mce_set_inter_base_multi_reference_penalty(0,
                                                                 mce_payload);
  intel_sub_group_avc_mce_set_inter_shape_penalty(0, mce_payload);
  intel_sub_group_avc_mce_set_inter_direction_penalty(0, mce_payload);
  intel_sub_group_avc_mce_set_motion_vector_cost_function(0, 0, 0, mce_payload);

  intel_sub_group_avc_mce_get_inter_reference_interlaced_field_polarities(
      0, 0, mce_result);
}

// CHECK-DAG: OpCapability Groups
// CHECK-DAG: OpCapability SubgroupAvcMotionEstimationINTEL
// CHECK-DAG: OpCapability SubgroupAvcMotionEstimationIntraINTEL
// CHECK-DAG: OpCapability SubgroupAvcMotionEstimationChromaINTEL
// CHECK-DAG: OpExtension "SPV_INTEL_device_side_avc_motion_estimation"


// CHECK: %[[#ImePayloadTy:]] = OpTypeAvcImePayloadINTEL
// CHECK: %[[#ImeSRefOutTy:]] = OpTypeAvcImeResultSingleReferenceStreamoutINTEL
// CHECK: %[[#ImeDRefOutTy:]] = OpTypeAvcImeResultDualReferenceStreamoutINTEL
// CHECK: %[[#ImeResultTy:]] = OpTypeAvcImeResultINTEL
// CHECK: %[[#MceResultTy:]] = OpTypeAvcMceResultINTEL
// CHECK: %[[#RefPayloadTy:]] = OpTypeAvcRefPayloadINTEL
// CHECK: %[[#SicPayloadTy:]] = OpTypeAvcSicPayloadINTEL
// CHECK: %[[#SicResultTy:]] = OpTypeAvcSicResultINTEL
// CHECK: %[[#McePayloadTy:]] = OpTypeAvcMcePayloadINTEL
// CHECK: %[[#ImeSRefInTy:]] = OpTypeAvcImeSingleReferenceStreaminINTEL
// CHECK: %[[#ImeDRefInTy:]] = OpTypeAvcImeDualReferenceStreaminINTEL

// CHECK: %[[#ImePayload:]] = OpFunctionParameter %[[#ImePayloadTy]]
// CHECK: %[[#ImeSRefOut:]] = OpFunctionParameter %[[#ImeSRefOutTy]]
// CHECK: %[[#ImeDRefOut:]] = OpFunctionParameter %[[#ImeDRefOutTy]]
// CHECK: %[[#ImeResult:]] = OpFunctionParameter %[[#ImeResultTy]]
// CHECK: %[[#MceResult:]] = OpFunctionParameter %[[#MceResultTy]]
// CHECK: %[[#RefPayload:]] = OpFunctionParameter %[[#RefPayloadTy]]
// CHECK: %[[#SicPayload:]] = OpFunctionParameter %[[#SicPayloadTy]]
// CHECK: %[[#SicResult:]] = OpFunctionParameter %[[#SicResultTy]]
// CHECK: %[[#McePayload:]] = OpFunctionParameter %[[#McePayloadTy]]


// CHECK: OpSubgroupAvcMceGetDefaultInterBaseMultiReferencePenaltyINTEL
// CHECK: OpSubgroupAvcMceGetDefaultInterShapePenaltyINTEL
// CHECK: OpSubgroupAvcMceGetDefaultIntraLumaShapePenaltyINTEL
// CHECK: OpSubgroupAvcMceGetDefaultInterMotionVectorCostTableINTEL
// CHECK: OpSubgroupAvcMceGetDefaultInterDirectionPenaltyINTEL
// CHECK: OpSubgroupAvcMceGetDefaultIntraLumaModePenaltyINTEL

// CHECK: OpSubgroupAvcImeInitializeINTEL %[[#ImePayloadTy]]
// CHECK: OpSubgroupAvcImeSetSingleReferenceINTEL %[[#ImePayloadTy]] {{.*}} %[[#ImePayload]]
// CHECK: OpSubgroupAvcImeSetDualReferenceINTEL %[[#ImePayloadTy]] {{.*}} %[[#ImePayload]]
// CHECK: OpSubgroupAvcImeRefWindowSizeINTEL
// CHECK: OpSubgroupAvcImeRefWindowSizeINTEL
// CHECK: OpSubgroupAvcImeAdjustRefOffsetINTEL
// CHECK: OpSubgroupAvcImeSetMaxMotionVectorCountINTEL %[[#ImePayloadTy]] {{.*}} %[[#ImePayload]]
// CHECK: OpSubgroupAvcImeGetSingleReferenceStreaminINTEL %[[#ImeSRefInTy]]{{.*}}%[[#ImeSRefOut]]
// CHECK: OpSubgroupAvcImeGetDualReferenceStreaminINTEL %[[#ImeDRefInTy]]{{.*}}%[[#ImeDRefOut]]
// CHECK: OpSubgroupAvcImeGetBorderReachedINTEL {{.*}} %[[#ImeResult]]
// CHECK: OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeDistortionsINTEL {{.*}} %[[#ImeSRefOut]]
// CHECK: OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeDistortionsINTEL {{.*}} %[[#ImeDRefOut]]
// CHECK: OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeMotionVectorsINTEL {{.*}} %[[#ImeSRefOut]]
// CHECK: OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeMotionVectorsINTEL {{.*}} %[[#ImeDRefOut]]
// CHECK: OpSubgroupAvcImeGetStreamoutSingleReferenceMajorShapeReferenceIdsINTEL {{.*}} %[[#ImeSRefOut]]
// CHECK: OpSubgroupAvcImeGetStreamoutDualReferenceMajorShapeReferenceIdsINTEL {{.*}} %[[#ImeDRefOut]]
// CHECK: OpSubgroupAvcImeSetDualReferenceINTEL %[[#ImePayloadTy]] {{.*}} %[[#ImePayload]]
// CHECK: OpSubgroupAvcImeSetWeightedSadINTEL %[[#ImePayloadTy]] {{.*}} %[[#ImePayload]]
// CHECK: OpSubgroupAvcImeSetEarlySearchTerminationThresholdINTEL %[[#ImePayloadTy]] {{.*}} %[[#ImePayload]]
// CHECK: OpSubgroupAvcFmeInitializeINTEL %[[#RefPayloadTy]]
// CHECK: OpSubgroupAvcBmeInitializeINTEL %[[#RefPayloadTy]]

// CHECK: OpSubgroupAvcRefSetBidirectionalMixDisableINTEL %[[#RefPayloadTy]]{{.*}}%[[#RefPayload]]

// CHECK: OpSubgroupAvcSicInitializeINTEL %[[#SicPayloadTy]]
// CHECK: OpSubgroupAvcSicConfigureIpeLumaINTEL %[[#SicPayloadTy]] {{.*}} %[[#SicPayload]]
// CHECK: OpSubgroupAvcSicConfigureIpeLumaChromaINTEL %[[#SicPayloadTy]] {{.*}} %[[#SicPayload]]
// CHECK: OpSubgroupAvcSicConfigureSkcINTEL %[[#SicPayloadTy]] {{.*}} %[[#SicPayload]]
// CHECK: OpSubgroupAvcSicSetSkcForwardTransformEnableINTEL %[[#SicPayloadTy]] {{.*}} %[[#SicPayload]]
// CHECK: OpSubgroupAvcSicSetBlockBasedRawSkipSadINTEL %[[#SicPayloadTy]] {{.*}} %[[#SicPayload]]
// CHECK: OpSubgroupAvcSicSetIntraLumaShapePenaltyINTEL %[[#SicPayloadTy]] {{.*}} %[[#SicPayload]]
// CHECK: OpSubgroupAvcSicSetIntraLumaModeCostFunctionINTEL %[[#SicPayloadTy]] {{.*}} %[[#SicPayload]]
// CHECK: OpSubgroupAvcSicSetIntraChromaModeCostFunctionINTEL %[[#SicPayloadTy]] {{.*}} %[[#SicPayload]]
// CHECK: OpSubgroupAvcSicGetBestIpeLumaDistortionINTEL {{.*}} %[[#SicResult]]
// CHECK: OpSubgroupAvcSicGetMotionVectorMaskINTEL

// CHECK: OpSubgroupAvcMceSetSourceInterlacedFieldPolarityINTEL %[[#McePayloadTy]] {{.*}} %[[#McePayload]]
// CHECK: OpSubgroupAvcMceSetSingleReferenceInterlacedFieldPolarityINTEL %[[#McePayloadTy]] {{.*}} %[[#McePayload]]
// CHECK: OpSubgroupAvcMceSetDualReferenceInterlacedFieldPolaritiesINTEL %[[#McePayloadTy]] {{.*}} %[[#McePayload]]
// CHECK: OpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL %[[#McePayloadTy]] {{.*}} %[[#McePayload]]
// CHECK: OpSubgroupAvcMceSetInterShapePenaltyINTEL %[[#McePayloadTy]] {{.*}} %[[#McePayload]]
// CHECK: OpSubgroupAvcMceSetInterDirectionPenaltyINTEL %[[#McePayloadTy]] {{.*}} %[[#McePayload]]
// CHECK: OpSubgroupAvcMceSetMotionVectorCostFunctionINTEL %[[#McePayloadTy]] {{.*}} %[[#McePayload]]
// CHECK: OpSubgroupAvcMceGetInterReferenceInterlacedFieldPolaritiesINTEL {{.*}} %[[#MceResult]]

