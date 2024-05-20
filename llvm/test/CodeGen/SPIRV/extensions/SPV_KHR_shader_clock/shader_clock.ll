; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_shader_clock %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_shader_clock %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: clock_read_device: the builtin requires the following SPIR-V extension: SPV_KHR_shader_clock

; CHECK: OpCapability ShaderClockKHR
; CHECK: OpExtension "SPV_KHR_shader_clock"
; CHECK-DAG: [[uint:%[a-z0-9_]+]] = OpTypeInt 32
; CHECK-DAG: [[ulong:%[a-z0-9_]+]] = OpTypeInt 64
; CHECK-DAG: [[v2uint:%[a-z0-9_]+]] = OpTypeVector [[uint]] 2
; CHECK-DAG: [[uint_1:%[a-z0-9_]+]] = OpConstant [[uint]] 1
; CHECK-DAG: [[uint_2:%[a-z0-9_]+]] = OpConstant [[uint]] 2
; CHECK-DAG: [[uint_3:%[a-z0-9_]+]] = OpConstant [[uint]] 3
; CHECK: OpReadClockKHR [[ulong]] [[uint_1]]
; CHECK: OpReadClockKHR [[ulong]] [[uint_2]]
; CHECK: OpReadClockKHR [[ulong]] [[uint_3]]
; CHECK: OpReadClockKHR [[v2uint]] [[uint_1]]
; CHECK: OpReadClockKHR [[v2uint]] [[uint_2]]
; CHECK: OpReadClockKHR [[v2uint]] [[uint_3]]

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @test_clocks(ptr addrspace(1) nocapture noundef writeonly align 8 %out64, ptr addrspace(1) nocapture noundef writeonly align 8 %outv2) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %call = tail call spir_func i64 @_Z17clock_read_devicev() #2
  store i64 %call, ptr addrspace(1) %out64, align 8, !tbaa !8
  %call1 = tail call spir_func i64 @_Z21clock_read_work_groupv() #2
  %arrayidx2 = getelementptr inbounds i8, ptr addrspace(1) %out64, i32 8
  store i64 %call1, ptr addrspace(1) %arrayidx2, align 8, !tbaa !8
  %call3 = tail call spir_func i64 @_Z20clock_read_sub_groupv() #2
  %arrayidx4 = getelementptr inbounds i8, ptr addrspace(1) %out64, i32 16
  store i64 %call3, ptr addrspace(1) %arrayidx4, align 8, !tbaa !8
  %call5 = tail call spir_func <2 x i32> @_Z22clock_read_hilo_devicev() #2
  store <2 x i32> %call5, ptr addrspace(1) %outv2, align 8, !tbaa !12
  %call7 = tail call spir_func <2 x i32> @_Z26clock_read_hilo_work_groupv() #2
  %arrayidx8 = getelementptr inbounds i8, ptr addrspace(1) %outv2, i32 8
  store <2 x i32> %call7, ptr addrspace(1) %arrayidx8, align 8, !tbaa !12
  %call9 = tail call spir_func <2 x i32> @_Z25clock_read_hilo_sub_groupv() #2
  %arrayidx10 = getelementptr inbounds i8, ptr addrspace(1) %outv2, i32 16
  store <2 x i32> %call9, ptr addrspace(1) %arrayidx10, align 8, !tbaa !12
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z17clock_read_devicev() local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z21clock_read_work_groupv() local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z20clock_read_sub_groupv() local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <2 x i32> @_Z22clock_read_hilo_devicev() local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <2 x i32> @_Z26clock_read_hilo_work_groupv() local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <2 x i32> @_Z25clock_read_hilo_sub_groupv() local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 19.0.0git"}
!3 = !{i32 1, i32 1}
!4 = !{!"none", !"none"}
!5 = !{!"ulong*", !"uint2*"}
!6 = !{!"ulong*", !"uint __attribute__((ext_vector_type(2)))*"}
!7 = !{!"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"long", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!10, !10, i64 0}
