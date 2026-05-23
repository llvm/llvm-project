; llvm/test/LTO/PowerPC/pragma-comment-copyright-thinlto.ll
;;
;; ThinLTO test for #pragma comment(copyright, ...) on AIX.
;;
;; Tests that copyright strings survive ThinLTO with the
;; ModuleSummaryAnalysis fix (findImplicitRefEdges) that adds
;; !implicit.ref-referenced globals as explicit reference edges
;; in the ThinLTO summary, keeping them live.
;;
;; f_add is notEligibleToImport because it references an internal
;; global (@__loadtime_comment_str) via !implicit.ref. Both copyright
;; strings still appear in the final binary via the external call path.

; REQUIRES: powerpc-registered-target

; RUN: rm -rf %t && mkdir %t
; RUN: split-file %s %t
; RUN: opt -passes='thinlto-pre-link<O2>' %t/tu1.ll -o - | \
; RUN:   opt -module-summary -o %t/tu1.bc
; RUN: opt -passes='thinlto-pre-link<O2>' %t/tu2.ll -o - | \
; RUN:   opt -module-summary -o %t/tu2.bc
; RUN: llvm-lto2 run -filetype=asm \
; RUN:   -r %t/tu1.bc,f_add,px \
; RUN:   -r %t/tu1.bc,f_unused,p \
; RUN:   -r %t/tu2.bc,main,px \
; RUN:   -r %t/tu2.bc,f_add,l \
; RUN:   %t/tu1.bc %t/tu2.bc -o %t/out
; RUN: FileCheck %s --check-prefix=CHECK-TU1 < %t/out.1
; RUN: FileCheck %s --check-prefix=CHECK-TU2 < %t/out.2

;--- tu1.ll
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

define i32 @f_add(i32 noundef %a, i32 noundef %b) {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

;; f_unused is not exported and never called -- ThinLTO must DCE it
define void @f_unused() {
entry:
  ret void
}

!comment_string.loadtime = !{!0}
!llvm.module.flags = !{!1, !2}
!0 = !{!"Copyright TU1"}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 1, !"EnableSplitLTOUnit", i32 0}

;--- tu2.ll
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

declare i32 @f_add(i32 noundef, i32 noundef)

define i32 @main() {
entry:
  %call = call i32 @f_add(i32 1, i32 2)
  ret i32 %call
}

!comment_string.loadtime = !{!0}
!llvm.module.flags = !{!1, !2}
!0 = !{!"Copyright TU2"}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 1, !"EnableSplitLTOUnit", i32 0}

;; TU1: f_add anchors copyright string, f_unused is DCE'd
; CHECK-TU1-LABEL: .f_add:
; CHECK-TU1-NEXT:  .ref __loadtime_comment_str
; CHECK-TU1-NOT:   .f_unused:
; CHECK-TU1:       .csect __loadtime_comment[RO]
; CHECK-TU1:       __loadtime_comment_str:
; CHECK-TU1-NEXT:  .string "Copyright TU1"

;; TU2: main anchors TU2 copyright string
;; f_add is NOT inlined -- notEligibleToImport because it references
;; an internal global via !implicit.ref. Cross-boundary call via
;; .extern .f_add keeps tu1 linked, preserving Copyright TU1.
; CHECK-TU2:      .ref __loadtime_comment_str
; CHECK-TU2:      bl .f_add
; CHECK-TU2:      .csect __loadtime_comment[RO]
; CHECK-TU2:      __loadtime_comment_str:
; CHECK-TU2-NEXT: .string "Copyright TU2"
; CHECK-TU2:      .extern .f_add
; CHECK-TU2-NOT:  .string "Copyright TU1"
