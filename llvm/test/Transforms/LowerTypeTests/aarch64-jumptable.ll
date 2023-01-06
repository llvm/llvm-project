; RUN: opt -S -passes=lowertypetests -mtriple=aarch64-unknown-linux-gnu %s | FileCheck --check-prefixes=AARCH64 %s

; Test for the jump table generation with branch protection on AArch64

target datalayout = "e-p:64:64"

@0 = private unnamed_addr constant [2 x ptr] [ptr @f, ptr @g], align 16

; AARCH64: @f = alias void (), ptr @[[JT:.*]]

define void @f() !type !0 {
  ret void
}

define internal void @g() !type !0 {
  ret void
}

!0 = !{i32 0, !"typeid1"}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

!llvm.module.flags = !{!1}

!1 = !{i32 4, !"branch-target-enforcement", i32 1}

; AARCH64:   define private void @[[JT]]() #[[ATTR:.*]] align 8 {

; AARCH64:      bti c
; AARCH64-SAME: b $0
; AARCH64-SAME: bti c
; AARCH64-SAME: b $1

; AARCH64: attributes #[[ATTR]] = { naked nounwind "branch-target-enforcement"="false" "sign-return-address"="none"
