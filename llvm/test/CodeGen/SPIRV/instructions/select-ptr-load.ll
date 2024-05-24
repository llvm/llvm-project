; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: %[[Float:.*]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[FloatPtr:.*]] = OpTypePointer Function %[[Float]]
; CHECK-SPIRV: OpInBoundsPtrAccessChain %[[FloatPtr]]
; CHECK-SPIRV: OpInBoundsPtrAccessChain %[[FloatPtr]]
; CHECK-SPIRV: OpSelect %[[FloatPtr]]
; CHECK-SPIRV: OpLoad %[[Float]]

%struct = type { [3 x float] }

define spir_kernel void @bar(i1 %sw) {
entry:
  %var1 = alloca %struct
  %var2 = alloca %struct
  %elem1 = getelementptr inbounds [3 x float], ptr %var1, i64 0, i64 0
  %elem2 = getelementptr inbounds [3 x float], ptr %var2, i64 0, i64 1
  %elem = select i1 %sw, ptr %elem1, ptr %elem2
  %res = load float, ptr %elem
  ret void
}
