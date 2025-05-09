// RUN: clang -cc1 -O1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-llvm %s -o %t.ll
// RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation %t.ll -o - | FileCheck %s

#pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable

void foo(__read_only image2d_t src, __read_only image2d_t ref,
         sampler_t sampler, intel_sub_group_avc_ime_payload_t ime_payload,
         intel_sub_group_avc_ime_single_reference_streamin_t sstreamin,
         intel_sub_group_avc_ime_dual_reference_streamin_t dstreamin,
         intel_sub_group_avc_ref_payload_t ref_payload,
         intel_sub_group_avc_sic_payload_t sic_payload) {
  intel_sub_group_avc_ime_evaluate_with_single_reference(src, ref, sampler,
                                                         ime_payload);
  intel_sub_group_avc_ime_evaluate_with_dual_reference(src, ref, ref, sampler,
                                                       ime_payload);
  intel_sub_group_avc_ime_evaluate_with_single_reference_streamout(
      src, ref, sampler, ime_payload);
  intel_sub_group_avc_ime_evaluate_with_dual_reference_streamout(
      src, ref, ref, sampler, ime_payload);
  intel_sub_group_avc_ime_evaluate_with_single_reference_streamin(
      src, ref, sampler, ime_payload, sstreamin);
  intel_sub_group_avc_ime_evaluate_with_dual_reference_streamin(
      src, ref, ref, sampler, ime_payload, dstreamin);
  intel_sub_group_avc_ime_evaluate_with_single_reference_streaminout(
      src, ref, sampler, ime_payload, sstreamin);
  intel_sub_group_avc_ime_evaluate_with_dual_reference_streaminout(
      src, ref, ref, sampler, ime_payload, dstreamin);

  intel_sub_group_avc_ref_evaluate_with_single_reference(src, ref, sampler,
                                                         ref_payload);
  intel_sub_group_avc_ref_evaluate_with_dual_reference(src, ref, ref, sampler,
                                                       ref_payload);
  intel_sub_group_avc_ref_evaluate_with_multi_reference(src, 0, sampler,
                                                        ref_payload);
  intel_sub_group_avc_ref_evaluate_with_multi_reference(src, 0, 0, sampler,
                                                        ref_payload);

  intel_sub_group_avc_sic_evaluate_with_single_reference(src, ref, sampler,
                                                         sic_payload);
  intel_sub_group_avc_sic_evaluate_with_dual_reference(src, ref, ref, sampler,
                                                       sic_payload);
  intel_sub_group_avc_sic_evaluate_with_multi_reference(src, 0, sampler,
                                                        sic_payload);
  intel_sub_group_avc_sic_evaluate_with_multi_reference(src, 0, 0, sampler,
                                                        sic_payload);
  intel_sub_group_avc_sic_evaluate_ipe(src, sampler, sic_payload);
}

// CHECK: OpCapability Groups
// CHECK: OpCapability SubgroupAvcMotionEstimationINTEL

// CHECK: OpExtension "SPV_INTEL_device_side_avc_motion_estimation"

// CHECK-DAG: %[[#ImageTy:]] = OpTypeImage
// CHECK-DAG: %[[#SamplerTy:]] = OpTypeSampler
// CHECK-DAG: %[[#ImePayloadTy:]] = OpTypeAvcImePayloadINTEL
// CHECK-DAG: %[[#ImeSRefInTy:]] = OpTypeAvcImeSingleReferenceStreaminINTEL
// CHECK-DAG: %[[#ImeDRefInTy:]] = OpTypeAvcImeDualReferenceStreaminINTEL
// CHECK-DAG: %[[#RefPayloadTy:]] = OpTypeAvcRefPayloadINTEL
// CHECK-DAG: %[[#SicPayloadTy:]] = OpTypeAvcSicPayloadINTEL
// CHECK-DAG: %[[#VmeImageTy:]] = OpTypeVmeImageINTEL
// CHECK-DAG: %[[#ImeResultTy:]] = OpTypeAvcImeResultINTEL
// CHECK-DAG: %[[#ImeSRefOutTy:]] = OpTypeAvcImeResultSingleReferenceStreamoutINTEL
// CHECK-DAG: %[[#ImeDRefOutTy:]] = OpTypeAvcImeResultDualReferenceStreamoutINTEL
// CHECK-DAG: %[[#RefResultTy:]] = OpTypeAvcRefResultINTEL
// CHECK-DAG: %[[#SicResultTy:]] = OpTypeAvcSicResultINTEL

// CHECK: %[[#SrcImg:]] = OpFunctionParameter %[[#ImageTy]]
// CHECK: %[[#RefImg:]] = OpFunctionParameter %[[#ImageTy]]
// CHECK: %[[#Sampler:]] = OpFunctionParameter %[[#SamplerTy]]
// CHECK: %[[#ImePayload:]] = OpFunctionParameter %[[#ImePayloadTy]]
// CHECK: %[[#ImeSRefIn:]] = OpFunctionParameter %[[#ImeSRefInTy]]
// CHECK: %[[#ImeDRefIn:]] = OpFunctionParameter %[[#ImeDRefInTy]]
// CHECK: %[[#RefPayload:]] = OpFunctionParameter %[[#RefPayloadTy]]
// CHECK: %[[#SicPayload:]] = OpFunctionParameter %[[#SicPayloadTy]]

// CHECK: %[[#VmeImg0:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg1:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcImeEvaluateWithSingleReferenceINTEL %[[#ImeResultTy]]{{.*}}%[[#VmeImg0]] %[[#VmeImg1]] %[[#ImePayload]]

// CHECK: %[[#VmeImg2:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg3:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg4:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcImeEvaluateWithDualReferenceINTEL %[[#ImeResultTy]]{{.*}}%[[#VmeImg2]] %[[#VmeImg3]] %[[#VmeImg4]] %[[#ImePayload]]

// CHECK: %[[#VmeImg5:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg6:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL %[[#ImeSRefOutTy]]{{.*}}%[[#VmeImg5]] %[[#VmeImg6]] %[[#ImePayload]]

// CHECK: %[[#VmeImg7:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg8:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg9:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL %[[#ImeDRefOutTy]]{{.*}}%[[#VmeImg7]] %[[#VmeImg8]] %[[#VmeImg9]] %[[#ImePayload]]

// CHECK: %[[#VmeImg10:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg11:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL %[[#ImeResultTy]]{{.*}}%[[#VmeImg10]] %[[#VmeImg11]] %[[#ImePayload]] %[[#ImeSRefIn]]

// CHECK: %[[#VmeImg12:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg13:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg14:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL %[[#ImeResultTy]]{{.*}}%[[#VmeImg12]] %[[#VmeImg13]] %[[#VmeImg14]] %[[#ImePayload]] %[[#ImeDRefIn]]

// CHECK: %[[#VmeImg1:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg2:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcImeEvaluateWithSingleReferenceStreaminoutINTEL %[[#ImeSRefOutTy]]{{.*}}%[[#VmeImg1]] %[[#VmeImg2]] %[[#ImePayload]] %[[#ImeSRefIn]]

// CHECK: %[[#VmeImg1:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg2:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg3:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcImeEvaluateWithDualReferenceStreaminoutINTEL %[[#ImeDRefOutTy]]{{.*}}%[[#VmeImg1]] %[[#VmeImg2]] %[[#VmeImg3]] %[[#ImePayload]] %[[#ImeDRefIn]]

// CHECK: %[[#VmeImg15:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg16:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcRefEvaluateWithSingleReferenceINTEL %[[#RefResultTy]]{{.*}}%[[#VmeImg15]] %[[#VmeImg16]] %[[#RefPayload]]

// CHECK: %[[#VmeImg17:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg18:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg19:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcRefEvaluateWithDualReferenceINTEL %[[#RefResultTy]]{{.*}}%[[#VmeImg17]] %[[#VmeImg18]] %[[#VmeImg19]] %[[#RefPayload]]


// CHECK: %[[#VmeImg20:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcRefEvaluateWithMultiReferenceINTEL %[[#RefResultTy]]{{.*}}%[[#VmeImg20]]{{.*}}%[[#RefPayload]]

// CHECK: %[[#VmeImg21:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL %[[#RefResultTy]]{{.*}}%[[#VmeImg21]]{{.*}}%[[#RefPayload]]

// CHECK: %[[#VmeImg23:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg24:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcSicEvaluateWithSingleReferenceINTEL %[[#SicResultTy]]{{.*}}%[[#VmeImg23]] %[[#VmeImg24]] %[[#SicPayload]]

// CHECK: %[[#VmeImg25:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg26:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: %[[#VmeImg27:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#RefImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcSicEvaluateWithDualReferenceINTEL %[[#SicResultTy]]{{.*}}%[[#VmeImg25]] %[[#VmeImg26]] %[[#VmeImg27]] %[[#SicPayload]]

// CHECK: %[[#VmeImg28:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcSicEvaluateWithMultiReferenceINTEL %[[#SicResultTy]]{{.*}}%[[#VmeImg28]]{{.*}}%[[#SicPayload]]

// CHECK: %[[#VmeImg29:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL %[[#SicResultTy]]{{.*}}%[[#VmeImg29]]{{.*}}%[[#SicPayload]]

// CHECK: %[[#VmeImg30:]] = OpVmeImageINTEL %[[#VmeImageTy]] %[[#SrcImg]] %[[#Sampler]]
// CHECK: OpSubgroupAvcSicEvaluateIpeINTEL %[[#SicResultTy]]{{.*}}%[[#VmeImg30]] %[[#SicPayload]]

