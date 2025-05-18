; RUN:llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_device_side_avc_motion_estimation %s -o - | FileCheck %s

; CHECK: OpCapability Groups
; CHECK: OpCapability SubgroupAvcMotionEstimationINTEL
; CHECK: OpExtension "SPV_INTEL_device_side_avc_motion_estimation"

; CHECK: OpTypeAvcMcePayloadINTEL
; CHECK: %[[#IME_PAYLOAD:]] = OpTypeAvcImePayloadINTEL
; CHECK: %[[#REF_PAYLOAD:]] = OpTypeAvcRefPayloadINTEL
; CHECK: %[[#SIC_PAYLOAD:]] = OpTypeAvcSicPayloadINTEL
; CHECK: OpTypeAvcMceResultINTEL
; CHECK: %[[#IME_RESULT:]] = OpTypeAvcImeResultINTEL
; CHECK: %[[#REF_RESULT:]] = OpTypeAvcRefResultINTEL
; CHECK: %[[#SIC_RESULT:]] = OpTypeAvcSicResultINTEL
; CHECK: %[[#SSTREAMOUT:]] =  OpTypeAvcImeResultSingleReferenceStreamoutINTEL
; CHECK: %[[#DSTREAMOUT:]] =  OpTypeAvcImeResultDualReferenceStreamoutINTEL
; CHECK: %[[#SSTREAMIN:]] =  OpTypeAvcImeSingleReferenceStreaminINTEL
; CHECK: %[[#DSTREAMIN:]] =  OpTypeAvcImeDualReferenceStreaminINTEL

; //CHECK: %[[#IME_PAYLOAD:]] = OpConstantNull
; //CHECK: %[[#REF_PAYLOAD:]] = OpConstantNull
; //CHECK: %[[#SIC_PAYLOAD:]] = OpConstantNull
; //CHECK: %[[#IME_RESULT:]] = OpConstantNull
; //CHECK: %[[#REF_RESULT:]] = OpConstantNull
; //CHECK: %[[#SIC_RESULT:]] = OpConstantNull
; //CHECK: %[[#SSTREAMOUT:]] = OpConstantNull
; //CHECK: %[[#DSTREAMOUT:]] = OpConstantNull
; //CHECK: %[[#SSTREAMIN:]] = OpConstantNull
; //CHECK: %[[#DSTREAMIN:]] = OpConstantNull

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

; Function Attrs: noinline nounwind optnone
define spir_func void @foo() #0 {
entry:
  %payload_mce = alloca target("spirv.AvcMcePayloadINTEL"), align 4
  %payload_ime = alloca target("spirv.AvcImePayloadINTEL"), align 4
  %payload_ref = alloca target("spirv.AvcRefPayloadINTEL"), align 4
  %payload_sic = alloca target("spirv.AvcSicPayloadINTEL"), align 4
  %result_mce = alloca target("spirv.AvcMceResultINTEL"), align 4
  %result_ime = alloca target("spirv.AvcImeResultINTEL"), align 4
  %result_ref = alloca target("spirv.AvcRefResultINTEL"), align 4
  %result_sic = alloca target("spirv.AvcSicResultINTEL"), align 4
  %sstreamout = alloca target("spirv.AvcImeResultSingleReferenceStreamoutINTEL"), align 4
  %dstreamout = alloca target("spirv.AvcImeResultDualReferenceStreamoutINTEL"), align 4
  %sstreamin = alloca target("spirv.AvcImeSingleReferenceStreaminINTEL"), align 4
  %dstreamin = alloca target("spirv.AvcImeDualReferenceStreaminINTEL"), align 4
  store target("spirv.AvcMcePayloadINTEL") zeroinitializer, ptr %payload_mce, align 4
  store target("spirv.AvcImePayloadINTEL") zeroinitializer, ptr %payload_ime, align 4
  store target("spirv.AvcRefPayloadINTEL") zeroinitializer, ptr %payload_ref, align 4
  store target("spirv.AvcSicPayloadINTEL") zeroinitializer, ptr %payload_sic, align 4
  store target("spirv.AvcMceResultINTEL") zeroinitializer, ptr %result_mce, align 4
  store target("spirv.AvcImeResultINTEL") zeroinitializer, ptr %result_ime, align 4
  store target("spirv.AvcRefResultINTEL") zeroinitializer, ptr %result_ref, align 4
  store target("spirv.AvcSicResultINTEL") zeroinitializer, ptr %result_sic, align 4
  store target("spirv.AvcImeResultSingleReferenceStreamoutINTEL") zeroinitializer, ptr %sstreamout, align 4
  store target("spirv.AvcImeResultDualReferenceStreamoutINTEL") zeroinitializer, ptr %dstreamout, align 4
  store target("spirv.AvcImeSingleReferenceStreaminINTEL") zeroinitializer, ptr %sstreamin, align 4
  store target("spirv.AvcImeDualReferenceStreaminINTEL") zeroinitializer, ptr %dstreamin, align 4
  ret void
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{}
!3 = !{!"clang version 5.0.1 (cfe/trunk)"}

