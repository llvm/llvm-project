; RUN: llc -mtriple=spirv32-unknown-unknown --spirv-extensions=SPV_KHR_expect_assume < %s | FileCheck --check-prefixes=EXT,CHECK %s
; RUN: llc -mtriple=spirv64-unknown-unknown --spirv-extensions=SPV_KHR_expect_assume < %s | FileCheck --check-prefixes=EXT,CHECK %s
; RUN: llc -mtriple=spirv32-unknown-unknown < %s | FileCheck --check-prefixes=NOEXT,CHECK %s
; RUN: llc -mtriple=spirv64-unknown-unknown < %s | FileCheck --check-prefixes=NOEXT,CHECK %s

; EXT:        OpCapability ExpectAssumeKHR
; EXT-NEXT:   OpExtension "SPV_KHR_expect_assume"
; NOEXT-NOT:  OpCapability ExpectAssumeKHR
; NOEXT-NOT:  OpExtension "SPV_KHR_expect_assume"

declare void @llvm.assume(i1)

; CHECK-DAG:  %9 = OpIEqual %5 %6 %7
; EXT-NEXT:   OpAssumeTrueKHR %9
; NOEXT-NOT:  OpAssumeTrueKHR %9
define void @assumeeq(i32 %x, i32 %y) {
    %cmp = icmp eq i32 %x, %y
    call void @llvm.assume(i1 %cmp)
    ret void
}
