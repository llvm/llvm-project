; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=SPV
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s --mattr=+spirv1.3 --spirv-extensions=SPV_KHR_float_controls -o - | FileCheck %s --check-prefixes=SPVEXT

define dso_local dllexport spir_kernel void @k_float_controls_0(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_1(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_2(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_3(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}

define dso_local dllexport spir_kernel void @k_float_controls_4(i32 %ibuf, i32 %obuf) local_unnamed_addr {
entry:
  ret void
}


!spirv.ExecutionMode = !{!15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29}

; SPV-NOT: OpExtension "SPV_KHR_float_controls"
; SPVEXT: OpExtension "SPV_KHR_float_controls"

; SPV-DAG: OpEntryPoint {{.*}} %[[#KERNEL0:]] "k_float_controls_0"
; SPV-DAG: OpEntryPoint {{.*}} %[[#KERNEL1:]] "k_float_controls_1"
; SPV-DAG: OpEntryPoint {{.*}} %[[#KERNEL2:]] "k_float_controls_2"
; SPV-DAG: OpEntryPoint {{.*}} %[[#KERNEL3:]] "k_float_controls_3"
; SPV-DAG: OpEntryPoint {{.*}} %[[#KERNEL4:]] "k_float_controls_4"

; SPV-DAG: OpExecutionMode %[[#KERNEL0]] DenormPreserve 64
!15 = !{void (i32, i32)* @k_float_controls_0, i32 4459, i32 64}
; SPV-DAG: OpExecutionMode %[[#KERNEL0]] DenormPreserve 32
!16 = !{void (i32, i32)* @k_float_controls_0, i32 4459, i32 32}
; SPV-DAG: OpExecutionMode %[[#KERNEL0]] DenormPreserve 16
!17 = !{void (i32, i32)* @k_float_controls_0, i32 4459, i32 16}

; SPV-DAG: OpExecutionMode %[[#KERNEL1]] DenormFlushToZero 64
!18 = !{void (i32, i32)* @k_float_controls_1, i32 4460, i32 64}
; SPV-DAG: OpExecutionMode %[[#KERNEL1]] DenormFlushToZero 32
!19 = !{void (i32, i32)* @k_float_controls_1, i32 4460, i32 32}
; SPV-DAG: OpExecutionMode %[[#KERNEL1]] DenormFlushToZero 16
!20 = !{void (i32, i32)* @k_float_controls_1, i32 4460, i32 16}

; SPV-DAG: OpExecutionMode %[[#KERNEL2]] SignedZeroInfNanPreserve 64
!21 = !{void (i32, i32)* @k_float_controls_2, i32 4461, i32 64}
; SPV-DAG: OpExecutionMode %[[#KERNEL2]] SignedZeroInfNanPreserve 32
!22 = !{void (i32, i32)* @k_float_controls_2, i32 4461, i32 32}
; SPV-DAG: OpExecutionMode %[[#KERNEL2]] SignedZeroInfNanPreserve 16
!23 = !{void (i32, i32)* @k_float_controls_2, i32 4461, i32 16}

; SPV-DAG: OpExecutionMode %[[#KERNEL3]] RoundingModeRTE 64
!24 = !{void (i32, i32)* @k_float_controls_3, i32 4462, i32 64}
; SPV-DAG: OpExecutionMode %[[#KERNEL3]] RoundingModeRTE 32
!25 = !{void (i32, i32)* @k_float_controls_3, i32 4462, i32 32}
; SPV-DAG: OpExecutionMode %[[#KERNEL3]] RoundingModeRTE 16
!26 = !{void (i32, i32)* @k_float_controls_3, i32 4462, i32 16}

; SPV-DAG: OpExecutionMode %[[#KERNEL4]] RoundingModeRTZ 64
!27 = !{void (i32, i32)* @k_float_controls_4, i32 4463, i32 64}
; SPV-DAG: OpExecutionMode %[[#KERNEL4]] RoundingModeRTZ 32
!28 = !{void (i32, i32)* @k_float_controls_4, i32 4463, i32 32}
; SPV-DAG: OpExecutionMode %[[#KERNEL4]] RoundingModeRTZ 16
!29 = !{void (i32, i32)* @k_float_controls_4, i32 4463, i32 16}
