; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_shader_clock %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_shader_clock %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: clock_read_device: the builtin requires the following SPIR-V extension: SPV_KHR_shader_clock

; CHECK: OpCapability ShaderClockKHR
; CHECK: OpExtension "SPV_KHR_shader_clock"
; CHECK-DAG: [[uint:%[a-z0-9_]+]] = OpTypeInt 32 0
; CHECK-DAG: [[ulong:%[a-z0-9_]+]] = OpTypeInt 64
; CHECK-DAG: [[v2uint:%[a-z0-9_]+]] = OpTypeVector [[uint]] 2
; CHECK-DAG: OpConstant [[uint]] 8
; CHECK-DAG: OpConstant [[uint]] 16
; CHECK-DAG: [[uint_1:%[a-z0-9_]+]] = OpConstant [[uint]] 1
; CHECK-DAG: [[uint_2:%[a-z0-9_]+]] = OpConstant [[uint]] 2
; CHECK-DAG: [[uint_3:%[a-z0-9_]+]] = OpConstant [[uint]] 3
; CHECK: OpReadClockKHR [[ulong]] [[uint_1]]
; CHECK: OpReadClockKHR [[ulong]] [[uint_2]]
; CHECK: OpReadClockKHR [[ulong]] [[uint_3]]
; CHECK: OpReadClockKHR [[v2uint]] [[uint_1]]
; CHECK: OpReadClockKHR [[v2uint]] [[uint_2]]
; CHECK: OpReadClockKHR [[v2uint]] [[uint_3]]

define dso_local spir_kernel void @test_clocks(ptr addrspace(1) nocapture noundef writeonly align 8 %out64, ptr addrspace(1) nocapture noundef writeonly align 8 %outv2) {
entry:
  %call = tail call spir_func i64 @_Z17clock_read_devicev()
  store i64 %call, ptr addrspace(1) %out64, align 8
  %call1 = tail call spir_func i64 @_Z21clock_read_work_groupv()
  %arrayidx2 = getelementptr inbounds i8, ptr addrspace(1) %out64, i32 8
  store i64 %call1, ptr addrspace(1) %arrayidx2, align 8
  %call3 = tail call spir_func i64 @_Z20clock_read_sub_groupv()
  %arrayidx4 = getelementptr inbounds i8, ptr addrspace(1) %out64, i32 16
  store i64 %call3, ptr addrspace(1) %arrayidx4, align 8
  %call5 = tail call spir_func <2 x i32> @_Z22clock_read_hilo_devicev()
  store <2 x i32> %call5, ptr addrspace(1) %outv2, align 8
  %call7 = tail call spir_func <2 x i32> @_Z26clock_read_hilo_work_groupv()
  %arrayidx8 = getelementptr inbounds i8, ptr addrspace(1) %outv2, i32 8
  store <2 x i32> %call7, ptr addrspace(1) %arrayidx8, align 8
  %call9 = tail call spir_func <2 x i32> @_Z25clock_read_hilo_sub_groupv()
  %arrayidx10 = getelementptr inbounds i8, ptr addrspace(1) %outv2, i32 16
  store <2 x i32> %call9, ptr addrspace(1) %arrayidx10, align 8
  ret void
}

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z17clock_read_devicev() local_unnamed_addr

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z21clock_read_work_groupv() local_unnamed_addr

; Function Attrs: convergent nounwind
declare spir_func i64 @_Z20clock_read_sub_groupv() local_unnamed_addr

; Function Attrs: convergent nounwind
declare spir_func <2 x i32> @_Z22clock_read_hilo_devicev() local_unnamed_addr

; Function Attrs: convergent nounwind
declare spir_func <2 x i32> @_Z26clock_read_hilo_work_groupv() local_unnamed_addr

; Function Attrs: convergent nounwind
declare spir_func <2 x i32> @_Z25clock_read_hilo_sub_groupv() local_unnamed_addr
