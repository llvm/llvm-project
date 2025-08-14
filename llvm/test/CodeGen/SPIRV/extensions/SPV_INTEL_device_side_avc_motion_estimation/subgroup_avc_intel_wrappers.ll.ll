; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation %s -o - | FileCheck %s
; XFAIL: *

; CHECK: OpCapability Groups
; CHECK: OpCapability SubgroupAvcMotionEstimationINTEL
; CHECK: OpExtension "SPV_INTEL_device_side_avc_motion_estimation"

; CHECK-DAG: %[[#ImePayloadTy:]] = OpTypeAvcImePayloadINTEL
; CHECK-DAG: %[[#ImeResultTy:]] = OpTypeAvcImeResultINTEL
; CHECK-DAG: %[[#RefPayloadTy:]] = OpTypeAvcRefPayloadINTEL
; CHECK-DAG: %[[#RefResultTy:]] = OpTypeAvcRefResultINTEL
; CHECK-DAG: %[[#SicPayloadTy:]] = OpTypeAvcSicPayloadINTEL
; CHECK-DAG: %[[#SicResultTy:]] = OpTypeAvcSicResultINTEL
; CHECK-DAG: %[[#McePayloadTy:]] = OpTypeAvcMcePayloadINTEL
; CHECK-DAG: %[[#MceResultTy:]] = OpTypeAvcMceResultINTEL

define spir_func void @test() #0 {
entry:
  %ime_payload = alloca target("spirv.AvcImePayloadINTEL"), align 8
  %ime_result = alloca target("spirv.AvcImeResultINTEL"), align 8
  %ref_payload = alloca target("spirv.AvcRefPayloadINTEL"), align 8
  %ref_result = alloca target("spirv.AvcRefResultINTEL"), align 8
  %sic_payload = alloca target("spirv.AvcSicPayloadINTEL"), align 8
  %sic_result = alloca target("spirv.AvcSicResultINTEL"), align 8

; CHECK: %[[#ImePayload:]] = OpLoad %[[#ImePayloadTy]]
; CHECK: %[[#ImeResult:]] = OpLoad %[[#ImeResultTy]]
; CHECK: %[[#RefPayload:]] = OpLoad %[[#RefPayloadTy]]
; CHECK: %[[#RefResult:]] = OpLoad %[[#RefResultTy]]
; CHECK: %[[#SicPayload:]] = OpLoad %[[#SicPayloadTy]]
; CHECK: %[[#SicResult:]] = OpLoad %[[#SicResultTy]]

  %0 = load target("spirv.AvcImePayloadINTEL"), target("spirv.AvcImePayloadINTEL")* %ime_payload, align 8
  %1 = load target("spirv.AvcImeResultINTEL"), target("spirv.AvcImeResultINTEL")* %ime_result, align 8
  %2 = load target("spirv.AvcRefPayloadINTEL"), target("spirv.AvcRefPayloadINTEL")* %ref_payload, align 8
  %3 = load target("spirv.AvcRefResultINTEL"), target("spirv.AvcRefResultINTEL")* %ref_result, align 8
  %4 = load target("spirv.AvcSicPayloadINTEL"), target("spirv.AvcSicPayloadINTEL")* %sic_payload, align 8
  %5 = load target("spirv.AvcSicResultINTEL"), target("spirv.AvcSicResultINTEL")* %sic_result, align 8

; CHECK:      %[[#ImeMcePayloadConv:]] = OpSubgroupAvcImeConvertToMcePayloadINTEL
; CHECK-SAME:     %[[#McePayloadTy]] %[[#ImePayload]]
; CHECK:      %[[#McePayloadRet0:]] = OpSubgroupAvcMceSetInterBaseMultiReferencePenaltyINTEL
; CHECK-SAME:     %[[#McePayloadTy]] {{.*}} %[[#ImeMcePayloadConv]]
; CHECK:      OpSubgroupAvcMceConvertToImePayloadINTEL
; CHECK-SAME:     %[[#ImePayloadTy]] %[[#McePayloadRet0]]
  %call0 = call spir_func target("spirv.AvcImePayloadINTEL") @_Z62intel_sub_group_avc_ime_set_inter_base_multi_reference_penaltyh37ocl_intel_sub_group_avc_ime_payload_t(i8 zeroext 0, target("spirv.AvcImePayloadINTEL") %0) #2

; CHECK:      %[[#ImeMceResultConv:]] = OpSubgroupAvcImeConvertToMceResultINTEL
; CHECK-SAME:     %[[#MceResultTy]] %[[#ImeResult]]
; CHECK:      OpSubgroupAvcMceGetMotionVectorsINTEL {{.*}} %[[#ImeMceResultConv]]
  %call1 = call spir_func i64 @_Z42intel_sub_group_avc_ime_get_motion_vectors36ocl_intel_sub_group_avc_ime_result_t(target("spirv.AvcImeResultINTEL") %1) #2

; CHECK:      %[[#RefMcePayloadConv:]] = OpSubgroupAvcRefConvertToMcePayloadINTEL
; CHECK-SAME:     %[[#McePayloadTy]] %[[#RefPayload]]
; CHECK:      %[[#McePayloadRet1:]] = OpSubgroupAvcMceSetInterShapePenaltyINTEL
; CHECK-SAME:     %[[#McePayloadTy]] {{.*}} %[[#RefMcePayloadConv]]
; CHECK:      OpSubgroupAvcMceConvertToRefPayloadINTEL
; CHECK-SAME:     %[[#RefPayloadTy]] %[[#McePayloadRet1]]
  %call2 = call spir_func target("spirv.AvcRefPayloadINTEL") @_Z47intel_sub_group_avc_ref_set_inter_shape_penaltym37ocl_intel_sub_group_avc_ref_payload_t(i64 0, target("spirv.AvcRefPayloadINTEL") %2) #2

; CHECK:      %[[#RefMceResultConv:]] = OpSubgroupAvcRefConvertToMceResultINTEL
; CHECK-SAME:     %[[#MceResultTy]] %[[#RefResult]]
; CHECK:      OpSubgroupAvcMceGetInterDistortionsINTEL {{.*}} %[[#RefMceResultConv]]
  %call3 = call spir_func zeroext i16 @_Z45intel_sub_group_avc_ref_get_inter_distortions36ocl_intel_sub_group_avc_ref_result_t(target("spirv.AvcRefResultINTEL") %3) #2

; CHECK:      %[[#SicMcePayloadConv:]] = OpSubgroupAvcSicConvertToMcePayloadINTEL
; CHECK-SAME:     %[[#McePayloadTy]] %[[#SicPayload]]
; CHECK:      %[[#McePayloadRet2:]] = OpSubgroupAvcMceSetMotionVectorCostFunctionINTEL
; CHECK-SAME:     %[[#McePayloadTy]] {{.*}} %[[#SicMcePayloadConv]]
; CHECK:      OpSubgroupAvcMceConvertToSicPayloadINTEL
; CHECK-SAME:     %[[#SicPayloadTy]] %[[#McePayloadRet2]]
  %call4 = call spir_func target("spirv.AvcSicPayloadINTEL") @_Z55intel_sub_group_avc_sic_set_motion_vector_cost_functionmDv2_jh37ocl_intel_sub_group_avc_sic_payload_t(i64 0, <2 x i32> zeroinitializer, i8 zeroext 0, target("spirv.AvcSicPayloadINTEL") %4) #2

; CHECK:      %[[#SicMceResultConv:]] = OpSubgroupAvcSicConvertToMceResultINTEL
; CHECK-SAME:     %[[#MceResultTy]] %[[#SicResult]]
; CHECK:      OpSubgroupAvcMceGetInterDistortionsINTEL {{.*}} %[[#SicMceResultConv]]
  %call5 = call spir_func zeroext i16 @_Z45intel_sub_group_avc_sic_get_inter_distortions36ocl_intel_sub_group_avc_sic_result_t(target("spirv.AvcSicResultINTEL") %5) #2
  ret void
}

declare spir_func target("spirv.AvcImePayloadINTEL") @_Z62intel_sub_group_avc_ime_set_inter_base_multi_reference_penaltyh37ocl_intel_sub_group_avc_ime_payload_t(i8 zeroext, target("spirv.AvcImePayloadINTEL")) #1
declare spir_func i64 @_Z42intel_sub_group_avc_ime_get_motion_vectors36ocl_intel_sub_group_avc_ime_result_t(target("spirv.AvcImeResultINTEL")) #1
declare spir_func target("spirv.AvcRefPayloadINTEL") @_Z47intel_sub_group_avc_ref_set_inter_shape_penaltym37ocl_intel_sub_group_avc_ref_payload_t(i64, target("spirv.AvcRefPayloadINTEL")) #1
declare spir_func zeroext i16 @_Z45intel_sub_group_avc_ref_get_inter_distortions36ocl_intel_sub_group_avc_ref_result_t(target("spirv.AvcRefResultINTEL")) #1
declare spir_func target("spirv.AvcSicPayloadINTEL") @_Z55intel_sub_group_avc_sic_set_motion_vector_cost_functionmDv2_jh37ocl_intel_sub_group_avc_sic_payload_t(i64, <2 x i32>, i8 zeroext, target("spirv.AvcSicPayloadINTEL")) #1
declare spir_func zeroext i16 @_Z45intel_sub_group_avc_sic_get_inter_distortions36ocl_intel_sub_group_avc_sic_result_t(target("spirv.AvcSicResultINTEL")) #1
