; Test that functions with "section" attribute are accepted, and jumptables are
; emitted in ".text".

; RUN: opt -S -passes=lowertypetests %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: @f = alias void (), ptr @[[JT:.*]]
; CHECK: define hidden void @f.cfi() section "xxx"

define void @f() section "xxx" !type !0 {
entry:
  ret void
}

define i1 @g() {
entry:
  %0 = call i1 @llvm.type.test(ptr @f, metadata !"_ZTSFvE")
  ret i1 %0
}

; CHECK: define private void @[[JT]]() #{{.*}} align {{.*}} {

declare i1 @llvm.type.test(ptr, metadata) nounwind readnone

!0 = !{i64 0, !"_ZTSFvE"}
