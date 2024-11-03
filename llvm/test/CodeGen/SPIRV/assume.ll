; RUN: llc -mtriple=spirv32-unknown-unknown --spirv-extensions=SPV_KHR_expect_assume < %s | FileCheck --check-prefixes=EXT,CHECK %s
; RUN: llc -mtriple=spirv64-unknown-unknown --spirv-extensions=SPV_KHR_expect_assume < %s | FileCheck --check-prefixes=EXT,CHECK %s
; RUN: llc -mtriple=spirv32-unknown-unknown < %s | FileCheck --check-prefixes=NOEXT,CHECK %s
; RUN: llc -mtriple=spirv64-unknown-unknown < %s | FileCheck --check-prefixes=NOEXT,CHECK %s

; EXT:        OpCapability ExpectAssumeKHR
; EXT-NEXT:   OpExtension "SPV_KHR_expect_assume"
; NOEXT-NOT:  OpCapability ExpectAssumeKHR
; NOEXT-NOT:  OpExtension "SPV_KHR_expect_assume"

declare void @llvm.assume(i1)

; CHECK-DAG:  %8 = OpIEqual %3 %5 %6
; EXT:        OpAssumeTrueKHR %8
; NOEXT-NOT:  OpAssumeTrueKHR %8
define i1 @assumeeq(i32 %x, i32 %y) {
    %cmp = icmp eq i32 %x, %y
    call void @llvm.assume(i1 %cmp)
    ret i1 %cmp
}
