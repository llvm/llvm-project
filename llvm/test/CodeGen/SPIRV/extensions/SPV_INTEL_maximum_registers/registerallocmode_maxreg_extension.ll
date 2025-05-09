; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_maximum_registers -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_INTEL_maximum_registers -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability RegisterLimitsINTEL
; CHECK: OpExtension "SPV_INTEL_maximum_registers"

; CHECK: OpMemoryModel Physical64 OpenCL
; CHECK: OpExecutionMode %[[#f1:]] MaximumRegistersINTEL 2
; CHECK: OpExecutionMode %[[#f2:]] MaximumRegistersINTEL 1
; CHECK: OpExecutionMode %[[#f3:]] NamedMaximumRegistersINTEL AutoINTEL
; CHECK: OpExecutionModeId %[[#f4:]] MaximumRegistersIdINTEL %[[#const_3:]]

; CHECK: %[[#void_type:]] = OpTypeVoid
; CHECK: %[[#func_type:]] = OpTypeFunction %[[#void_type]]

; CHECK: %[[#int_type:]] = OpTypeInt 32 0
; CHECK: %[[#const_3]] = OpConstant %[[#int_type]] 3

; CHECK: %[[#f1]] = OpFunction %[[#void_type]] DontInline %[[#]]
; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l3() #0 !RegisterAllocMode !10 {
newFuncRoot:
  ret void
}

; CHECK: %[[#f2]] = OpFunction %[[#void_type]] DontInline %[[#]]
; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l6() #0 !RegisterAllocMode !11 {
newFuncRoot:
  ret void
}

; CHECK: %[[#f3]] = OpFunction %[[#void_type]] DontInline %[[#]]
; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l9() #0 !RegisterAllocMode !12 {
newFuncRoot:
  ret void
}

; CHECK: %[[#f4]] = OpFunction %[[#void_type]] DontInline %[[#]]
; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l13() #0 !RegisterAllocMode !13 {
newFuncRoot:
  ret void
}

; CHECK: %[[#f5:]] = OpFunction %[[#void_type]] DontInline %[[#]]
; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @main_l19() #0 {
newFuncRoot:
  ret void
}

attributes #0 = { noinline nounwind optnone }


!opencl.compiler.options = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!2, !3, !3, !3, !3, !3, !2, !3, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!llvm.module.flags = !{!4, !5, !6, !7, !8}
!spirv.MemoryModel = !{!9, !9, !9, !9, !9, !9}
!spirv.ExecutionMode = !{}

!0 = !{}
!2 = !{i32 4, i32 200000}
!3 = !{i32 3, i32 200000}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"openmp", i32 50}
!6 = !{i32 7, !"openmp-device", i32 50}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{i32 2, i32 2}
!10 = !{i32 2}
!11 = !{i32 1}
!12 = !{!"AutoINTEL"}
!13 = !{!14}
!14 = !{i32 3}

