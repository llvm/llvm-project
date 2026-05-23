;; ThinLTO test for #pragma comment(copyright, ...).
;; Tests that Prgam commentcopyright strings from module TU1 gets imported 
;; to module TU2 and is gets added to TU2's @llvm.compiler.used.

; REQUIRES: powerpc-registered-target

; RUN: rm -rf %t && mkdir %t
; RUN: split-file %s %t
; RUN: opt -passes='thinlto-pre-link<O2>' %t/tu1.ll -o - | \
; RUN:   opt -module-summary -o %t/tu1.bc
; RUN: opt -passes='thinlto-pre-link<O2>' %t/tu2.ll -o - | \
; RUN:   opt -module-summary -o %t/tu2.bc
; RUN: llvm-lto --thinlto-action=thinlink -o combined %t/tu1.bc %t/tu2.bc
; RUN: llvm-lto --thinlto-action=import  \
; RUN:         --thinlto-index=combined  \
; RUN:         --exported-symbol=main    \
; RUN:         --exported-symbol=f_add   \
; RUN:         --exported-symbol=my_function \
; RUN:        %t/tu2.bc -o %t/tu2.imported.bc
; RUN: llvm-dis %t/tu2.imported.bc -o - | FileCheck %s --check-prefix=CHECK-TU2-IMPORTED

;--- tu1.ll
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

define i32 @f_add(i32 noundef %a, i32 noundef %b) {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

!comment_string.loadtime = !{!0}
!llvm.module.flags = !{!1, !2}
!0 = !{!"@(#) Copyright TU1"}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 1, !"EnableSplitLTOUnit", i32 0}

;--- tu2.ll
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

declare i32 @f_add(i32 noundef, i32 noundef)

define i32 @main() {
entry:
  %call = tail call i32 @f_add(i32 noundef 1, i32 noundef 2)
  ret i32 %call
}

!comment_string.loadtime = !{!0}
!llvm.module.flags = !{!1, !2}
!0 = !{!"@(#) Copyright TU2"}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 1, !"EnableSplitLTOUnit", i32 0}

; CHECK-TU2-IMPORTED: @[[TU2_STR:__loadtime_comment_str_[0-9a-f]+]] = weak_odr hidden unnamed_addr constant [19 x i8] c"@(#) Copyright TU2\00", section "__loadtime_comment", align 1
; CHECK-TU2-IMPORTED-NEXT: @[[TU1_STR:__loadtime_comment_str_[0-9a-f]+]] = available_externally hidden unnamed_addr constant [19 x i8] c"@(#) Copyright TU1\00", section "__loadtime_comment", align 1
; CHECK-TU2-IMPORTED-NEXT: @llvm.compiler.used = appending global [2 x ptr] [ptr @[[TU2_STR]], ptr @[[TU1_STR]]], section "llvm.metadata"

; CHECK-TU2-IMPORTED-LABEL: define i32 @main()
; CHECK-TU2-IMPORTED-SAME: local_unnamed_addr !implicit.ref ![[MAIN_MD:[0-9]+]]

; CHECK-TU2-IMPORTED-LABEL: define available_externally i32 @f_add(
; CHECK-TU2-IMPORTED-SAME: local_unnamed_addr #[[FADD_ATTR:[0-9]+]] !implicit.ref ![[FADD_MD:[0-9]+]]

; CHECK-TU2-IMPORTED: ![[MAIN_MD]] = !{ptr @[[TU2_STR]]}
; CHECK-TU2-IMPORTED: ![[FADD_MD]] = !{ptr @[[TU1_STR]]}
