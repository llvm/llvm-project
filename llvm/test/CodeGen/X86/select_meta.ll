; RUN: opt -mtriple=x86_64-unknown-unknown -codegenprepare -S < %s 2>&1 | FileCheck %s

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @foo(i32, i32, i32) {
  %4 = and i32 %0, 3
  %5 = icmp eq i32 %4, 1
  %6 = select i1 %5, i32 %1, i32 %2, !prof  !1
; CHECK: br {{.*}}label{{.*}}, label{{.*}}, !prof ![[WT:.*]]
  ret i32 %6
}

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (trunk 279683)"}
!1 = !{!"branch_weights", i32 1000, i32 1 }
; CHECK: ![[WT]] = !{!"branch_weights", i32 1000, i32 1}
