; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpName [[ADD:%.*]] "test_add"
; CHECK-DAG: OpName [[SUB:%.*]] "test_sub"
; CHECK-DAG: OpName [[MIN:%.*]] "test_min"
; CHECK-DAG: OpName [[MAX:%.*]] "test_max"
; CHECK-DAG: OpName [[UMIN:%.*]] "test_umin"
; CHECK-DAG: OpName [[UMAX:%.*]] "test_umax"
; CHECK-DAG: OpName [[AND:%.*]] "test_and"
; CHECK-DAG: OpName [[OR:%.*]] "test_or"
; CHECK-DAG: OpName [[XOR:%.*]] "test_xor"

; CHECK-DAG: [[I32Ty:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[PtrI32Ty:%.*]] = OpTypePointer Function [[I32Ty]]
; CHECK-DAG: [[I64Ty:%.*]] = OpTypeInt 64 0
; CHECK-DAG: [[PtrI64Ty:%.*]] = OpTypePointer Generic [[I64Ty]]
; CHECK-DAG: [[CROSSDEVICESCOPE:%.*]] = OpConstantNull [[I32Ty]]
; CHECK-DAG: [[DEVICESCOPE:%.*]] = OpConstant [[I32Ty]] 1
;; "monotonic" maps to the relaxed memory semantics, encoded with constant 0

; CHECK:      [[ADD]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicIAdd [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_add(i32* %ptr, i32 %val) {
  %r = atomicrmw add i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK:      [[SUB]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicISub [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_sub(i32* %ptr, i32 %val) {
  %r = atomicrmw sub i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK:      [[MIN]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicSMin [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_min(i32* %ptr, i32 %val) {
  %r = atomicrmw min i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK:      [[MAX]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicSMax [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_max(i32* %ptr, i32 %val) {
  %r = atomicrmw max i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK:      [[UMIN]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicUMin [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_umin(i32* %ptr, i32 %val) {
  %r = atomicrmw umin i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK:      [[UMAX]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicUMax [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_umax(i32* %ptr, i32 %val) {
  %r = atomicrmw umax i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK:      [[AND]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicAnd [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_and(i32* %ptr, i32 %val) {
  %r = atomicrmw and i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK:      [[OR]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicOr [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_or(i32* %ptr, i32 %val) {
  %r = atomicrmw or i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK:      [[XOR]] = OpFunction [[I32Ty]]
; CHECK-NEXT: [[A:%.*]] = OpFunctionParameter [[PtrI32Ty]]
; CHECK-NEXT: [[B:%.*]] = OpFunctionParameter [[I32Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[R:%.*]] = OpAtomicXor [[I32Ty]] [[A]] [[CROSSDEVICESCOPE]] {{.+}} [[B]]
; CHECK-NEXT: OpReturnValue [[R]]
; CHECK-NEXT: OpFunctionEnd
define i32 @test_xor(i32* %ptr, i32 %val) {
  %r = atomicrmw xor i32* %ptr, i32 %val monotonic
  ret i32 %r
}

; CHECK: OpFunction
; CHECK-NEXT: [[Arg1:%.*]] = OpFunctionParameter [[PtrI64Ty]]
; CHECK-NEXT: [[Arg2:%.*]] = OpFunctionParameter [[I64Ty]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpAtomicSMin [[I64Ty]] [[Arg1]] [[DEVICESCOPE]] {{.+}} [[Arg2]]
; CHECK-NEXT: OpAtomicSMax [[I64Ty]] [[Arg1]] [[DEVICESCOPE]] {{.+}} [[Arg2]]
; CHECK-NEXT: OpAtomicUMin [[I64Ty]] [[Arg1]] [[DEVICESCOPE]] {{.+}} [[Arg2]]
; CHECK-NEXT: OpAtomicUMax [[I64Ty]] [[Arg1]] [[DEVICESCOPE]] {{.+}} [[Arg2]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define dso_local spir_kernel void @test_wrappers(ptr addrspace(4) %arg, i64 %val) {
  %r1 = call spir_func i64 @__spirv_AtomicSMin(ptr addrspace(4) %arg, i32 1, i32 0, i64 %val)
  %r2 = call spir_func i64 @__spirv_AtomicSMax(ptr addrspace(4) %arg, i32 1, i32 0, i64 %val)
  %r3 = call spir_func i64 @__spirv_AtomicUMin(ptr addrspace(4) %arg, i32 1, i32 0, i64 %val)
  %r4 = call spir_func i64 @__spirv_AtomicUMax(ptr addrspace(4) %arg, i32 1, i32 0, i64 %val)
  ret void
}

declare dso_local spir_func i64 @__spirv_AtomicSMin(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @__spirv_AtomicSMax(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @__spirv_AtomicUMin(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @__spirv_AtomicUMax(ptr addrspace(4), i32, i32, i64)
