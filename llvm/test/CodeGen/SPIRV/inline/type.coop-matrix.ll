; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s

; TODO: enable spirv-val once we can add cooperative matrix capability and extension

; CHECK: [[float_t:%[0-9]+]] = OpTypeFloat 32
; CHECK: [[uint32_t:%[0-9]+]] = OpTypeInt 32 0

; CHECK: [[uint32_2:%[0-9]+]] = OpConstant [[uint32_t]] 2
%scope = type target("spirv.IntegralConstant", i32, 2) ; Workgroup
; CHECK: [[uint32_4:%[0-9]+]] = OpConstant [[uint32_t]] 4
%cols = type target("spirv.IntegralConstant", i32, 4)
%rows = type target("spirv.IntegralConstant", i32, 4)
; CHECK: [[uint32_0:%[0-9]+]] = OpConstant [[uint32_t]] 0
%use = type target("spirv.IntegralConstant", i32, 0) ; MatrixAKHR

; CHECK: OpUnknown(4456, 7) [[coop_t:%[0-9]+]] [[float_t]] [[uint32_2]] [[uint32_4]] [[uint32_4]] [[uint32_0]]
%coop_t = type target("spirv.Type", float, %scope, %rows, %cols, %use, 4456, 0, 0)

; CHECK: [[getCooperativeMatrix_t:%[0-9]+]] = OpTypeFunction [[coop_t]]

; CHECK: [[getCooperativeMatrix:%[0-9]+]] = OpFunction [[coop_t]] None [[getCooperativeMatrix_t]]
declare %coop_t @getCooperativeMatrix()

define void @main() #1 {
entry:
; CHECK: {{%[0-9]+}} = OpFunctionCall [[coop_t]] [[getCooperativeMatrix]]
  %coop = call %coop_t @getCooperativeMatrix()

  ret void
}
