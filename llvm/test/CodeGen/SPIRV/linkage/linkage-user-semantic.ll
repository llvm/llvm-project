; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-use-user-semantic-for-linkage %s -o - | FileCheck %s --check-prefix=CHECK-ON
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-OFF

; FIXME: currently spirv-val is not erroring out when UserSemantic decoration
; is applied to a function, but it actually should.
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-use-user-semantic-for-linkage %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Test that --spirv-use-user-semantic-for-linkage emits UserSemantic decorations
; for LLVM linkage types with no native SPIR-V representation.

; CHECK-ON-DAG: OpName %[[#WEAK_FN:]] "weak_func"
; CHECK-ON-DAG: OpName %[[#WEAK:]] "weak_var"
; CHECK-ON-DAG: OpName %[[#WEAK_ODR:]] "weak_odr_var"
; CHECK-ON-DAG: OpName %[[#LINKONCE:]] "linkonce_var"
; CHECK-ON-DAG: OpName %[[#COMMON:]] "common_var"
; CHECK-ON-DAG: OpName %[[#EW:]] "ew_func"

; CHECK-ON-DAG: OpDecorate %[[#WEAK_FN]] UserSemantic "linkage:weak"
; CHECK-ON-DAG: OpDecorate %[[#WEAK]] UserSemantic "linkage:weak"
; CHECK-ON-DAG: OpDecorate %[[#WEAK_ODR]] UserSemantic "linkage:weak_odr"
; CHECK-ON-DAG: OpDecorate %[[#LINKONCE]] UserSemantic "linkage:linkonce"
; CHECK-ON-DAG: OpDecorate %[[#COMMON]] UserSemantic "linkage:common"
; CHECK-ON-DAG: OpDecorate %[[#EW]] UserSemantic "linkage:extern_weak"

; CHECK-OFF-NOT: UserSemantic "linkage:

@weak_var = weak addrspace(1) global i32 0, align 4
@weak_odr_var = weak_odr addrspace(1) global i32 0, align 4
@linkonce_var = linkonce addrspace(1) global i32 0, align 4
@common_var = common addrspace(1) global i32 0, align 4

declare extern_weak spir_func void @ew_func()

define weak spir_func i32 @weak_func(i32 %x) {
entry:
  ret i32 %x
}

define spir_func void @caller() {
entry:
  %val = call i32 @weak_func(i32 42)
  call spir_func void @ew_func()
  ret void
}
