; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

; Verify that a (mangled or non-mangled) call to the SPIR-V friendly OpenCL
; built-in `__spirv_AbortKHR` is lowered to OpAbortKHR.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#NM:]] = OpConstant %[[#I32]] 7
; CHECK-DAG: %[[#MN:]] = OpConstant %[[#I32]] 11
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I32]] %[[#I32]] %[[#I32]]

; CHECK: OpAbortKHR %[[#I32]] %[[#NM]]
; CHECK-NOT: OpUnreachable
; CHECK: OpAbortKHR %[[#I32]] %[[#MN]]
; CHECK-NOT: OpUnreachable

; The OpenCL C ABI passes aggregate arguments by pointer (byval). Verify that
; such an argument is loaded so OpAbortKHR receives the composite by value.
; CHECK: %[[#LOADED:]] = OpLoad %[[#STRUCT]]
; CHECK: OpAbortKHR %[[#STRUCT]] %[[#LOADED]]

%struct.Msg = type { i32, i32, i32 }

; Non-mangled SPIR-V friendly name (commonly used in OpenCL C/C++ via __spirv_*).
declare void @__spirv_AbortKHR(i32) #0

; Itanium-mangled OpenCL C name: __spirv_AbortKHR(unsigned int).
declare void @_Z16__spirv_AbortKHRj(i32) #0

; Itanium-mangled OpenCL C name with a struct argument; passed by pointer.
declare void @_Z16__spirv_AbortKHR3Msg(ptr byval(%struct.Msg)) #0

define spir_kernel void @kernel_nonmangled() {
entry:
  call void @__spirv_AbortKHR(i32 7)
  unreachable
}

define spir_kernel void @kernel_mangled() {
entry:
  call void @_Z16__spirv_AbortKHRj(i32 11)
  unreachable
}

define spir_kernel void @kernel_struct(i32 %x, i32 %y, i32 %z) {
entry:
  %m = alloca %struct.Msg, align 4
  %p0 = getelementptr inbounds %struct.Msg, ptr %m, i32 0, i32 0
  store i32 %x, ptr %p0, align 4
  %p1 = getelementptr inbounds %struct.Msg, ptr %m, i32 0, i32 1
  store i32 %y, ptr %p1, align 4
  %p2 = getelementptr inbounds %struct.Msg, ptr %m, i32 0, i32 2
  store i32 %z, ptr %p2, align 4
  call void @_Z16__spirv_AbortKHR3Msg(ptr byval(%struct.Msg) %m)
  unreachable
}

attributes #0 = { noreturn }
