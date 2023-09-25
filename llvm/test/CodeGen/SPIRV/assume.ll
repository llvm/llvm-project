; RUN: llc -mtriple=spirv32-unknown-unknown < %s | FileCheck %s
; RUN: llc -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; CHECK:      OpCapability ExpectAssumeKHR
; CHECK-NEXT: OpExtension "SPV_KHR_expect_assume"

declare void @llvm.assume(i1)

; CHECK-DAG:  %9 = OpIEqual %5 %6 %7
; CHECK-NEXT: OpAssumeTrueKHR %9
define void @assumeeq(i32 %x, i32 %y) {
    %cmp = icmp eq i32 %x, %y
    call void @llvm.assume(i1 %cmp)
    ret void
}
