; Test that functions with "section" attribute are accepted, and jumptables are
; emitted in ".text".

; RUN: opt -S -lowerbitsets < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: @[[A:.*]] = private constant {{.*}} section ".text"
; CHECK: @f = alias void (), bitcast ({{.*}}* @[[A]] to void ()*)
; CHECK: define private void {{.*}} section "xxx"

define void @f() section "xxx" {
entry:
  ret void
}

define i1 @g() {
entry:
  %0 = call i1 @llvm.bitset.test(i8* bitcast (void ()* @f to i8*), metadata !"_ZTSFvE")
  ret i1 %0
}

declare i1 @llvm.bitset.test(i8*, metadata) nounwind readnone

!llvm.bitsets = !{!0}
!0 = !{!"_ZTSFvE", void ()* @f, i64 0}
