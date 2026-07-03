;; Test verifies inlining happens cross module when module flags are upgraded.
;; `foo` and `main` are both old semantic while bar is the new semantic.
;; Regression test for #82763

; RUN: split-file %s %t
; RUN: opt -module-summary %t/foo.ll -o %t/foo.o
; RUN: opt -module-summary %t/bar.ll -o %t/bar.o
; RUN: opt -module-summary %t/main.ll -o %t/main.o
; RUN: llvm-lto2 run %t/main.o %t/foo.o %t/bar.o -save-temps \
; RUN:   -o %t/t.exe \
; RUN:   -r=%t/foo.o,foo,plx \
; RUN:   -r=%t/bar.o,bar,plx \
; RUN:   -r=%t/main.o,foo,l \
; RUN:   -r=%t/main.o,bar,l \
; RUN:   -r=%t/main.o,main,plx 2>&1
; RUN: llvm-dis %t/t.exe.1.4.opt.bc -o - | FileCheck %s

; CHECK:      define dso_local noundef i32 @main() local_unnamed_addr #0 {
; CHECK-NEXT: entry:
; CHECK-NEXT:  ret i32 35
; CHECK-NEXT: }

; CHECK:  attributes #0 = { {{.*}}"branch-target-enforcement" "sign-return-address"="all" "sign-return-address-key"="b_key" }

; CHECK: !llvm.module.flags = !{!0, !1, !2, !3}

; CHECK: !0 = !{i32 8, !"branch-target-enforcement", i32 2}
; CHECK: !1 = !{i32 8, !"sign-return-address", i32 2}
; CHECK: !2 = !{i32 8, !"sign-return-address-all", i32 2}
; CHECK: !3 = !{i32 8, !"sign-return-address-with-bkey", i32 2}


;--- foo.ll
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define dso_local noundef i32 @foo() local_unnamed_addr #0 {
entry:
  ret i32 34
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 1}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 1}

;--- bar.ll
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define dso_local noundef i32 @bar() local_unnamed_addr #0 {
entry:
  ret i32 1
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "branch-target-enforcement" "sign-return-address"="all" "sign-return-address-key"="b_key" }
!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 2}
!1 = !{i32 8, !"sign-return-address", i32 2}
!2 = !{i32 8, !"sign-return-address-all", i32 2}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 2}

;--- main.ll
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare i32 @foo();
declare i32 @bar();

define i32 @main() #0 {
entry:
  %1 = call i32 @foo()
  %2 = call i32 @bar()
  %3 = add i32 %1, %2
  ret i32 %3
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 1}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 1}
