; Source:
; #pragma OPENCL EXTENSION cl_intel_device_side_avc_motion_estimation : enable
;
; void foo(__read_only image2d_t src,
;          __read_only image2d_t ref,
;          sampler_t sampler) {
;   intel_sub_group_avc_ime_payload_t ime_payload;
;   intel_sub_group_avc_ime_single_reference_streamin_t sstreamin;
;   intel_sub_group_avc_ime_dual_reference_streamin_t dstreamin;
;
;   intel_sub_group_avc_ime_evaluate_with_single_reference(
;     src, ref, sampler, ime_payload);
;   intel_sub_group_avc_ime_evaluate_with_dual_reference(
;     src, ref, ref, sampler, ime_payload);
;   intel_sub_group_avc_ime_evaluate_with_single_reference_streamout(
;     src, ref, sampler, ime_payload);
;   intel_sub_group_avc_ime_evaluate_with_dual_reference_streamout(
;     src, ref, ref, sampler, ime_payload);
;   intel_sub_group_avc_ime_evaluate_with_single_reference_streamin(
;     src, ref, sampler, ime_payload, sstreamin);
;   intel_sub_group_avc_ime_evaluate_with_dual_reference_streamin(
;     src, ref, ref, sampler, ime_payload, dstreamin);
;
;   intel_sub_group_avc_ref_payload_t ref_payload;
;
;   intel_sub_group_avc_ref_evaluate_with_single_reference(
;     src, ref, sampler, ref_payload);
;   intel_sub_group_avc_ref_evaluate_with_dual_reference(
;     src, ref, ref, sampler, ref_payload);
;   intel_sub_group_avc_ref_evaluate_with_multi_reference(
;     src, 0, sampler, ref_payload);
;   intel_sub_group_avc_ref_evaluate_with_multi_reference(
;     src, 0, 0, sampler, ref_payload);
;
;   intel_sub_group_avc_sic_payload_t sic_payload;
;  intel_sub_group_avc_sic_evaluate_with_single_reference(
;     src, ref, sampler, sic_payload);
;   intel_sub_group_avc_sic_evaluate_with_dual_reference(
;     src, ref, ref, sampler, sic_payload);
;   intel_sub_group_avc_sic_evaluate_with_multi_reference(
;     src, 0, sampler, sic_payload);
;   intel_sub_group_avc_sic_evaluate_with_multi_reference(
;     src, 0, 0, sampler, sic_payload);
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s

; CHECK: Capability Groups
; CHECK: Capability SubgroupAvcMotionEstimationINTEL

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK: TypeImage              [[ImageTy:[0-9]+]]
; CHECK: TypeSampler            [[SamplerTy:[0-9]+]]
; CHECK: TypeAvcImePayloadINTEL [[ImePayloadTy:[0-9]+]]
; CHECK: TypeAvcImeSingleReferenceStreaminINTEL        [[ImeSRefInTy:[0-9]+]]
; CHECK: TypeAvcImeDualReferenceStreaminINTEL          [[ImeDRefInTy:[0-9]+]]
; CHECK: TypeAvcRefPayloadINTEL [[RefPayloadTy:[0-9]+]]
; CHECK: TypeAvcSicPayloadINTEL [[SicPayloadTy:[0-9]+]]
; CHECK: TypeVmeImageINTEL      [[VmeImageTy:[0-9]+]] [[ImageTy]]
; CHECK: TypeAvcImeResultINTEL  [[ImeResultTy:[0-9]+]]
; CHECK: TypeAvcImeResultSingleReferenceStreamoutINTEL [[ImeSRefOutTy:[0-9]+]]
; CHECK: TypeAvcImeResultDualReferenceStreamoutINTEL   [[ImeDRefOutTy:[0-9]+]]
; CHECK: TypeAvcRefResultINTEL  [[RefResultTy:[0-9]+]]
; CHECK: TypeAvcSicResultINTEL  [[SicResultTy:[0-9]+]]

%opencl.image2d_ro_t = type opaque
%opencl.sampler_t = type opaque
%opencl.intel_sub_group_avc_ime_payload_t = type opaque
%opencl.intel_sub_group_avc_ime_single_reference_streamin_t = type opaque
%opencl.intel_sub_group_avc_ime_dual_reference_streamin_t = type opaque
%opencl.intel_sub_group_avc_ref_payload_t = type opaque
%opencl.intel_sub_group_avc_sic_payload_t = type opaque
%opencl.intel_sub_group_avc_ime_result_t = type opaque
%opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t = type opaque
%opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t = type opaque
%opencl.intel_sub_group_avc_ref_result_t = type opaque
%opencl.intel_sub_group_avc_sic_result_t = type opaque

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @foo(%opencl.image2d_ro_t addrspace(1)* %src, %opencl.image2d_ro_t addrspace(1)* %ref, %opencl.sampler_t addrspace(2)* %sampler) #0 {
entry:
  %src.addr = alloca %opencl.image2d_ro_t addrspace(1)*, align 8
  %ref.addr = alloca %opencl.image2d_ro_t addrspace(1)*, align 8
  %sampler.addr = alloca %opencl.sampler_t addrspace(2)*, align 8
  %ime_payload = alloca %opencl.intel_sub_group_avc_ime_payload_t*, align 8
  %sstreamin = alloca %opencl.intel_sub_group_avc_ime_single_reference_streamin_t*, align 8
  %dstreamin = alloca %opencl.intel_sub_group_avc_ime_dual_reference_streamin_t*, align 8
  %ref_payload = alloca %opencl.intel_sub_group_avc_ref_payload_t*, align 8
  %sic_payload = alloca %opencl.intel_sub_group_avc_sic_payload_t*, align 8
  store %opencl.image2d_ro_t addrspace(1)* %src, %opencl.image2d_ro_t addrspace(1)** %src.addr, align 8
  store %opencl.image2d_ro_t addrspace(1)* %ref, %opencl.image2d_ro_t addrspace(1)** %ref.addr, align 8
  store %opencl.sampler_t addrspace(2)* %sampler, %opencl.sampler_t addrspace(2)** %sampler.addr, align 8

; CHECK: Load [[ImageTy]] [[Image0:[0-9]+]]
; CHECK: Load [[ImageTy]] [[Image1:[0-9]+]]
; CHECK: Load [[SamplerTy]] [[Sampler:[0-9]+]]
; CHECK: Load [[ImePayloadTy]] [[ImePayload:[0-9]+]]
; CHECK: Load [[ImeSRefInTy]] [[ImeSRefIn:[0-9]+]]
; CHECK: Load [[ImeDRefInTy]] [[ImeDRefIn:[0-9]+]]
; CHECK: Load [[RefPayloadTy]] [[RefPayload:[0-9]+]]
; CHECK: Load [[SicPayloadTy]] [[SicPayload:[0-9]+]]
  %0 = load %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)** %src.addr, align 8
  %1 = load %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)** %ref.addr, align 8
  %2 = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** %sampler.addr, align 8
  %3 = load %opencl.intel_sub_group_avc_ime_payload_t*, %opencl.intel_sub_group_avc_ime_payload_t** %ime_payload, align 8
  %4 = load %opencl.intel_sub_group_avc_ime_single_reference_streamin_t*, %opencl.intel_sub_group_avc_ime_single_reference_streamin_t** %sstreamin, align 8
  %5 = load %opencl.intel_sub_group_avc_ime_dual_reference_streamin_t*, %opencl.intel_sub_group_avc_ime_dual_reference_streamin_t** %dstreamin, align 8
  %6 = load %opencl.intel_sub_group_avc_ref_payload_t*, %opencl.intel_sub_group_avc_ref_payload_t** %ref_payload, align 8
  %7 = load %opencl.intel_sub_group_avc_sic_payload_t*, %opencl.intel_sub_group_avc_sic_payload_t** %sic_payload, align 8

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg0:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg1:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcImeEvaluateWithSingleReferenceINTEL [[ImeResultTy]] {{.*}} [[VmeImg0]] [[VmeImg1]] [[ImePayload]]
  %call = call spir_func %opencl.intel_sub_group_avc_ime_result_t* @_Z54intel_sub_group_avc_ime_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ime_payload_t* %3) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg2:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg3:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg4:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcImeEvaluateWithDualReferenceINTEL [[ImeResultTy]] {{.*}} [[VmeImg2]] [[VmeImg3]] [[VmeImg4]] [[ImePayload]]
  %call1 = call spir_func %opencl.intel_sub_group_avc_ime_result_t* @_Z52intel_sub_group_avc_ime_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ime_payload_t* %3) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg5:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg6:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcImeEvaluateWithSingleReferenceStreamoutINTEL [[ImeSRefOutTy]] {{.*}} [[VmeImg5]] [[VmeImg6]] [[ImePayload]]
  %call2 = call spir_func %opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t* @_Z64intel_sub_group_avc_ime_evaluate_with_single_reference_streamout14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ime_payload_t* %3) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg7:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg8:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg9:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcImeEvaluateWithDualReferenceStreamoutINTEL [[ImeDRefOutTy]] {{.*}} [[VmeImg7]] [[VmeImg8]] [[VmeImg9]] [[ImePayload]]
  %call3 = call spir_func %opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t* @_Z62intel_sub_group_avc_ime_evaluate_with_dual_reference_streamout14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ime_payload_t* %3) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg10:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg11:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcImeEvaluateWithSingleReferenceStreaminINTEL [[ImeResultTy]] {{.*}} [[VmeImg10]] [[VmeImg11]] [[ImePayload]] [[ImeSRefIn]]
  %call4 = call spir_func %opencl.intel_sub_group_avc_ime_result_t* @_Z63intel_sub_group_avc_ime_evaluate_with_single_reference_streamin14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t55ocl_intel_sub_group_avc_ime_single_reference_streamin_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ime_payload_t* %3, %opencl.intel_sub_group_avc_ime_single_reference_streamin_t* %4) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg12:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg13:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg14:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcImeEvaluateWithDualReferenceStreaminINTEL [[ImeResultTy]] {{.*}} [[VmeImg12]] [[VmeImg13]] [[VmeImg14]] [[ImePayload]] [[ImeDRefIn]]
  %call5 = call spir_func %opencl.intel_sub_group_avc_ime_result_t* @_Z61intel_sub_group_avc_ime_evaluate_with_dual_reference_streamin14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t53ocl_intel_sub_group_avc_ime_dual_reference_streamin_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ime_payload_t* %3, %opencl.intel_sub_group_avc_ime_dual_reference_streamin_t* %5) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg15:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg16:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcRefEvaluateWithSingleReferenceINTEL [[RefResultTy]] {{.*}} [[VmeImg15]] [[VmeImg16]] [[RefPayload]]
  %call6 = call spir_func %opencl.intel_sub_group_avc_ref_result_t* @_Z54intel_sub_group_avc_ref_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ref_payload_t* %6) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg17:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg18:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg19:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcRefEvaluateWithDualReferenceINTEL [[RefResultTy]] {{.*}} [[VmeImg17]] [[VmeImg18]] [[VmeImg19]] [[RefPayload]]
  %call7 = call spir_func %opencl.intel_sub_group_avc_ref_result_t* @_Z52intel_sub_group_avc_ref_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ref_payload_t* %6) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg20:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: SubgroupAvcRefEvaluateWithMultiReferenceINTEL [[RefResultTy]] {{.*}} [[VmeImg20]] {{.*}} [[RefPayload]]
  %call8 = call spir_func %opencl.intel_sub_group_avc_ref_result_t* @_Z53intel_sub_group_avc_ref_evaluate_with_multi_reference14ocl_image2d_roj11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, i32 0, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ref_payload_t* %6) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg21:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: SubgroupAvcRefEvaluateWithMultiReferenceInterlacedINTEL [[RefResultTy]] {{.*}} [[VmeImg21]] {{.*}} [[RefPayload]]
  %call9 = call spir_func %opencl.intel_sub_group_avc_ref_result_t* @_Z53intel_sub_group_avc_ref_evaluate_with_multi_reference14ocl_image2d_rojh11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, i32 0, i8 zeroext 0, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_ref_payload_t* %6) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg23:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg24:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcSicEvaluateWithSingleReferenceINTEL [[SicResultTy]] {{.*}} [[VmeImg23]] [[VmeImg24]] [[SicPayload]]
  %call10 = call spir_func %opencl.intel_sub_group_avc_sic_result_t* @_Z54intel_sub_group_avc_sic_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_sic_payload_t* %7) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg25:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg26:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg27:[0-9]+]] [[Image1]] [[Sampler]]
; CHECK: SubgroupAvcSicEvaluateWithDualReferenceINTEL [[SicResultTy]] {{.*}} [[VmeImg25]] [[VmeImg26]] [[VmeImg27]] [[SicPayload]]
  %call11 = call spir_func %opencl.intel_sub_group_avc_sic_result_t* @_Z52intel_sub_group_avc_sic_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.image2d_ro_t addrspace(1)* %1, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_sic_payload_t* %7) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg28:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: SubgroupAvcSicEvaluateWithMultiReferenceINTEL [[SicResultTy]] {{.*}} [[VmeImg28]] {{.*}} [[SicPayload]]
  %call12 = call spir_func %opencl.intel_sub_group_avc_sic_result_t* @_Z53intel_sub_group_avc_sic_evaluate_with_multi_reference14ocl_image2d_roj11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, i32 0, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_sic_payload_t* %7) #2

; CHECK: VmeImageINTEL [[VmeImageTy]] [[VmeImg29:[0-9]+]] [[Image0]] [[Sampler]]
; CHECK: SubgroupAvcSicEvaluateWithMultiReferenceInterlacedINTEL [[SicResultTy]] {{.*}} [[VmeImg29]] {{.*}} [[SicPayload]]
  %call13 = call spir_func %opencl.intel_sub_group_avc_sic_result_t* @_Z53intel_sub_group_avc_sic_evaluate_with_multi_reference14ocl_image2d_rojh11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%opencl.image2d_ro_t addrspace(1)* %0, i32 0, i8 zeroext 0, %opencl.sampler_t addrspace(2)* %2, %opencl.intel_sub_group_avc_sic_payload_t* %7) #2
  ret void
}

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_result_t* @_Z54intel_sub_group_avc_ime_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ime_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_result_t* @_Z52intel_sub_group_avc_ime_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ime_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_result_single_reference_streamout_t* @_Z64intel_sub_group_avc_ime_evaluate_with_single_reference_streamout14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ime_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_result_dual_reference_streamout_t* @_Z62intel_sub_group_avc_ime_evaluate_with_dual_reference_streamout14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ime_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_result_t* @_Z63intel_sub_group_avc_ime_evaluate_with_single_reference_streamin14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t55ocl_intel_sub_group_avc_ime_single_reference_streamin_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ime_payload_t*, %opencl.intel_sub_group_avc_ime_single_reference_streamin_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ime_result_t* @_Z61intel_sub_group_avc_ime_evaluate_with_dual_reference_streamin14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ime_payload_t53ocl_intel_sub_group_avc_ime_dual_reference_streamin_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ime_payload_t*, %opencl.intel_sub_group_avc_ime_dual_reference_streamin_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ref_result_t* @_Z54intel_sub_group_avc_ref_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ref_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ref_result_t* @_Z52intel_sub_group_avc_ref_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ref_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ref_result_t* @_Z53intel_sub_group_avc_ref_evaluate_with_multi_reference14ocl_image2d_roj11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%opencl.image2d_ro_t addrspace(1)*, i32, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ref_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_ref_result_t* @_Z53intel_sub_group_avc_ref_evaluate_with_multi_reference14ocl_image2d_rojh11ocl_sampler37ocl_intel_sub_group_avc_ref_payload_t(%opencl.image2d_ro_t addrspace(1)*, i32, i8 zeroext, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_ref_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_sic_result_t* @_Z54intel_sub_group_avc_sic_evaluate_with_single_reference14ocl_image2d_roS_11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_sic_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_sic_result_t* @_Z52intel_sub_group_avc_sic_evaluate_with_dual_reference14ocl_image2d_roS_S_11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.image2d_ro_t addrspace(1)*, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_sic_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_sic_result_t* @_Z53intel_sub_group_avc_sic_evaluate_with_multi_reference14ocl_image2d_roj11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%opencl.image2d_ro_t addrspace(1)*, i32, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_sic_payload_t*) #1

; Function Attrs: convergent
declare spir_func %opencl.intel_sub_group_avc_sic_result_t* @_Z53intel_sub_group_avc_sic_evaluate_with_multi_reference14ocl_image2d_rojh11ocl_sampler37ocl_intel_sub_group_avc_sic_payload_t(%opencl.image2d_ro_t addrspace(1)*, i32, i8 zeroext, %opencl.sampler_t addrspace(2)*, %opencl.intel_sub_group_avc_sic_payload_t*) #1

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
