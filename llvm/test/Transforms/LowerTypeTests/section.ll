; Test that functions with "section" attribute are accepted, and jumptables are
; emitted in ".text".

; RUN: opt -S -passes=lowertypetests %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: @f = alias [8 x i8], ptr @[[JT:.*]]
; CHECK: define hidden void @f.cfi() section "xxx"

define void @f() section "xxx" !type !0 {
entry:
  ret void
}

define i1 @g() !prof !1 {
entry:
  %0 = call i1 @llvm.type.test(ptr @f, metadata !"_ZTSFvE")
  ret i1 %0
}

define i1 @h(i1 %c) !prof !2 {
entry:
  br i1 %c, label %yes, label %common, !prof !3

yes:
  %0 = call i1 @llvm.type.test(ptr @f, metadata !"_ZTSFvE")
  ret i1 %0

common:
  ret i1 0
}

; CHECK: define private void @[[JT]]() #{{.*}} align {{.*}} !prof !4 {

declare i1 @llvm.type.test(ptr, metadata) nounwind readnone

!0 = !{i64 0, !"_ZTSFvE"}
!1 = !{!"function_entry_count", i32 20} 
!2 = !{!"function_entry_count", i32 40}
!3 = !{!"branch_weights", i32 3, i32 5}
; the entry count for the jumptable function is: 20 + 40 * (3/8) = 20 + 15
; where: 20 is the entry count of g, 40 of h, and 3/8 is the frequency of the
; llvm.type.test in h, relative to h's entry basic block.                               
; CHECK !4 = !{!"function_entry_count", i64 35}