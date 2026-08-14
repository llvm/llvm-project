; A function used as a data pointer (load/store through @fn) requires
; SPV_INTEL_function_pointers, since it needs OpTypePointer with an
; OpTypeFunction pointee.

; Without the extension: using a function as a data pointer is an error.
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: not llc -O2 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; With the extension: pointer-to-function types and FunctionPointerINTEL constants are emitted.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s --check-prefix=CHECK-EXT
; RUN: llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - | FileCheck %s --check-prefix=CHECK-EXT

; TODO: Update when spirv-val accepts OpTypePointer with an OpTypeFunction pointee.
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: error:{{.*}}Function used as a data pointer requires SPV_INTEL_function_pointers extension

; CHECK-EXT-DAG: OpCapability FunctionPointersINTEL
; CHECK-EXT-DAG: OpExtension "SPV_INTEL_function_pointers"
; CHECK-EXT-DAG: %[[#FUNCTY:]] = OpTypeFunction
; CHECK-EXT-DAG: %[[#PTR_CW:]] = OpTypePointer CrossWorkgroup %[[#FUNCTY]]
; CHECK-EXT-DAG: %[[#PTR_CS:]] = OpTypePointer CodeSectionINTEL %[[#FUNCTY]]
; CHECK-EXT-DAG: OpConstantFunctionPointerINTEL %[[#PTR_CS]]

define spir_kernel void @fuzz_kernel_load(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2) {
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %5, label %9
5:
  %6 = load i32, ptr @fuzz_kernel_load, align 4
  %7 = mul i32 %2, -1640531527
  %8 = add i32 %6, %6
  ret void
9:
  unreachable
}

define spir_kernel void @fuzz_kernel_store(ptr addrspace(1) nofree readnone captures(none) %in,
                                            ptr addrspace(1) nofree readnone captures(none) %out,
                                            i32 %n) local_unnamed_addr {
  %ok = icmp sgt i32 %n, 0
  br i1 %ok, label %1, label %2
1:
  store i32 -1640531527, ptr @fuzz_kernel_store, align 4
  br label %2
2:
  ret void
}
