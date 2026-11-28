; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_inline_assembly %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_inline_assembly %s -o - -filetype=obj | spirv-val %}

; An aggregate value reaching inline asm as a typed value-id used to crash type propagation.

%Inner = type { float, float }
%Outer = type { i32, %Inner }

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Inner:]] = OpTypeStruct %[[#Float]] %[[#Float]]
; CHECK-DAG: %[[#Const:]] = OpConstantComposite %[[#Inner]]

; CHECK-LABEL: Begin function load
; CHECK: %[[#L:]] = OpLoad %[[#Inner]]
; CHECK: OpAsmCallINTEL %[[#]] %[[#]] %[[#L]]
define i32 @load(ptr addrspace(1) %p) {
  %s = load %Inner, ptr addrspace(1) %p
  %r = call i32 asm sideeffect "", "=r,r"(%Inner %s)
  ret i32 %r
}

; CHECK-LABEL: Begin function constant
; CHECK: OpAsmCallINTEL %[[#]] %[[#]] %[[#Const]]
define i32 @constant() {
  %r = call i32 asm sideeffect "", "=r,r"(%Inner { float 1.0, float 2.0 })
  ret i32 %r
}

; CHECK-LABEL: Begin function extract_select
; CHECK: %[[#E:]] = OpCompositeExtract %[[#Inner]] %[[#]] 1
; CHECK: OpAsmCallINTEL %[[#]] %[[#]] %[[#E]]
; CHECK: OpSelect %[[#Inner]] %[[#]] %[[#E]]
define i32 @extract_select(%Outer %s, i1 %c) {
  %a = extractvalue %Outer %s, 1
  %r = call i32 asm sideeffect "", "=r,r"(%Inner %a)
  %p = select i1 %c, %Inner %a, %Inner zeroinitializer
  %x = extractvalue %Inner %p, 0
  %i = fptosi float %x to i32
  %y = add i32 %r, %i
  ret i32 %y
}

; CHECK-LABEL: Begin function select
; CHECK: %[[#S:]] = OpSelect %[[#Inner]]
; CHECK: OpAsmCallINTEL %[[#]] %[[#]] %[[#S]]
define i32 @select(i1 %c, %Inner %a, %Inner %b) {
  %s = select i1 %c, %Inner %a, %Inner %b
  %r = call i32 asm sideeffect "", "=r,r"(%Inner %s)
  ret i32 %r
}

; CHECK-LABEL: Begin function argument
; CHECK: %[[#A:]] = OpFunctionParameter %[[#Inner]]
; CHECK: OpAsmCallINTEL %[[#]] %[[#]] %[[#A]]
define i32 @argument(%Inner %a) {
  %r = call i32 asm sideeffect "", "=r,r"(%Inner %a)
  ret i32 %r
}
