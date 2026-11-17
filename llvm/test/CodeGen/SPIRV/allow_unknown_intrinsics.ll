; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spv-allow-unknown-intrinsics %s -o %t.spvt 2>&1 | FileCheck -check-prefix=CHECK-ERROR %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spv-allow-unknown-intrinsics=notllvm %s -o %t.spvt 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spv-allow-unknown-intrinsics=llvm.some.custom %s -o %t.spvt 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spv-allow-unknown-intrinsics=llvm. %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spv-allow-unknown-intrinsics=llvm.,random.prefix %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spv-allow-unknown-intrinsics=llvm. %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

; The test checks command-line option which allows to represent unknown
; intrinsics as external function calls in SPIR-V.

; CHECK-ERROR: LLVM ERROR: unable to legalize instruction: %3:iid(s64) = G_READCYCLECOUNTER (in function: foo)

; CHECK: Name %[[READCYCLECOUNTER:[0-9]+]] "spirv.llvm_readcyclecounter"
; CHECK: Name %[[SOME_CUSTOM_INTRINSIC:[0-9]+]] "spirv.llvm_some_custom_intrinsic"
; CHECK-DAG: Decorate %[[READCYCLECOUNTER]] LinkageAttributes {{.*}} Import
; CHECK: Decorate %[[SOME_CUSTOM_INTRINSIC]] LinkageAttributes {{.*}} Import
; CHECK-DAG: %[[I64:[0-9]+]] = OpTypeInt 64
; CHECK: %[[FnTy:[0-9]+]] = OpTypeFunction %[[I64]]
; CHECK: %[[READCYCLECOUNTER]] = OpFunction %[[I64]] {{.*}} %[[FnTy]]
; CHECK-DAG: %[[SOME_CUSTOM_INTRINSIC]] = OpFunction %[[I64]] {{.*}} %[[FnTy]]
; CHECK-DAG: OpFunctionCall %[[I64]] %[[READCYCLECOUNTER]]
; CHECK:     OpFunctionCall %[[I64]] %[[SOME_CUSTOM_INTRINSIC]]

define spir_func void @foo() {
entry:
; TODO: if and when the SPIR-V learns how to lower readcyclecounter, we will have to pick another unhandled intrinsic
  %0 = call i64 @llvm.readcyclecounter()
  %1 = call i64 @llvm.some.custom.intrinsic()
  ret void
}

declare i64 @llvm.readcyclecounter()
declare i64 @llvm.some.custom.intrinsic()
