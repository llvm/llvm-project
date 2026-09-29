; RUN: llc -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_expect_assume < %s | FileCheck --check-prefixes=EXT,CHECK %s
; RUN: llc -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_expect_assume < %s | FileCheck --check-prefixes=EXT,CHECK %s
; RUN: llc -mtriple=spirv32-unknown-unknown < %s | FileCheck --check-prefixes=NOEXT,CHECK %s
; RUN: llc -mtriple=spirv64-unknown-unknown < %s | FileCheck --check-prefixes=NOEXT,CHECK %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_expect_assume < %s -o - -filetype=obj | spirv-val %}

; EXT-DAG:    OpCapability ExpectAssumeKHR
; EXT-DAG:    OpExtension "SPV_KHR_expect_assume"
; NOEXT-NOT:  OpCapability ExpectAssumeKHR
; NOEXT-NOT:  OpExtension "SPV_KHR_expect_assume"

declare void @llvm.assume(i1)

; CHECK:      %[[#X:]] = OpFunctionParameter
; CHECK:      %[[#Y:]] = OpFunctionParameter
; CHECK:      %[[#CMP:]] = OpIEqual %[[#]] %[[#X]] %[[#Y]]
; EXT:        OpAssumeTrueKHR %[[#CMP]]
; NOEXT-NOT:  OpAssumeTrueKHR
define i1 @assumeeq(i32 %x, i32 %y) {
    %cmp = icmp eq i32 %x, %y
    call void @llvm.assume(i1 %cmp)
    ret i1 %cmp
}

; NOTE: The operand bundle information is not lowered to
; SPIR-V as there is no corresponding representation
; CHECK:      %[[#]] = OpFunction
; CHECK:      %[[#]] = OpFunctionParameter
; CHECK:      %[[#]] = OpLabel
; EXT:        OpAssumeTrueKHR %[[#]]
; NOEXT-NOT:  OpAssumeTrueKHR
; CHECK:      OpReturn
; CHECK:      OpFunctionEnd
define void @assume_with_operand_bundles(ptr %p) {
    call void @llvm.assume(i1 true) [ "align"(ptr %p, i64 64), "nonnull"(ptr %p) ]
    ret void
}
