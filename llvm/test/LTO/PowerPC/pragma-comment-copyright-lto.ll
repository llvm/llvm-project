;;
;; Full LTO test for #pragma comment(copyright, ...) on AIX.
;; Two TUs each with a copyright string are linked via Full LTO.
;;
;; Verifies:
;;   - LowerCommentStringPass runs at lto-pre-link<O2> per TU
;;   - f_unused is DCE'd -- not exported and never called
;;   - f_add is inlined into main and f_add(1,2) constant-folded to li 3,3
;;   - Both copyright strings appear in the merged __loadtime_comment[RO] csect
;;   - .ref directives anchor each string to its function csect

; REQUIRES: powerpc-registered-target

; RUN: rm -rf %t && mkdir %t
; RUN: split-file %s %t
; RUN: opt -passes='lto-pre-link<O2>' %t/tu1.ll -S -o %t/tu1_lowered.ll
; RUN: opt -passes='lto-pre-link<O2>' %t/tu2.ll -S -o %t/tu2_lowered.ll
; RUN: llvm-as %t/tu1_lowered.ll -o %t/tu1.bc
; RUN: llvm-as %t/tu2_lowered.ll -o %t/tu2.bc
; RUN: llvm-lto -filetype=asm \
; RUN:   -exported-symbol=main \
; RUN:   -exported-symbol=f_add \
; RUN:   %t/tu1.bc %t/tu2.bc \
; RUN:   -o %t/out.s
; RUN: FileCheck %s < %t/out.s

;--- tu1.ll
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

define i32 @f_add(i32 noundef %a, i32 noundef %b) {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

;; f_unused is not exported and never called -- Full LTO must DCE it
define void @f_unused() {
entry:
  ret void
}

!comment_string.loadtime = !{!0}
!llvm.module.flags = !{!1, !2}
!0 = !{!"Copyright TU1"}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 1, !"EnableSplitLTOUnit", i32 1}

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
!2 = !{i32 1, !"EnableSplitLTOUnit", i32 1}

;; f_add csect anchors TU1 copyright string.
; CHECK-LABEL: .f_add:
; CHECK-NEXT:  .ref [[TU1_STR:__loadtime_comment_str_[0-9a-f]+]]

;; main anchors both strings: TU2's own and TU1's via inlined f_add.
; CHECK-LABEL: .main:
; CHECK-DAG:   .ref [[TU1_STR]]
; CHECK-DAG:   .ref [[TU2_STR:__loadtime_comment_str_[0-9a-f]+]]

;; f_add(1,2) constant-folded -- no optimization regression with pragma.
; CHECK:       li 3, 3

;; Both copyright strings in the read-only __loadtime_comment csect.
; CHECK:       .csect __loadtime_comment[RO]
; CHECK-NEXT:  .lglobl [[TU1_STR]]
; CHECK:       [[TU1_STR]]:
; CHECK-NEXT:  .string "Copyright TU1"
; CHECK-NEXT:  .lglobl [[TU2_STR]]
; CHECK:       [[TU2_STR]]:
; CHECK-NEXT:  .string "Copyright TU2"

;; f_unused is DCE'd -- must not appear in assembly.
; CHECK-NOT:   .f_unused:
