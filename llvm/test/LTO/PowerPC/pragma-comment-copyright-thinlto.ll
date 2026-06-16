;; ThinLTO test for #pragma comment(copyright, ...).
;; Verifies that the copyright string from TU1 gets imported into TU2
;; and added to TU2's @llvm.compiler.used, with !implicit.ref attached
;; to both main and the imported f_tu1.

; REQUIRES: powerpc-registered-target

; RUN: rm -rf %t && mkdir %t
; RUN: split-file %s %t
; RUN: opt -passes='thinlto-pre-link<O2>' %t/tu1.ll -o - | \
; RUN:   opt -module-summary -o %t/tu1.bc
; RUN: opt -passes='thinlto-pre-link<O2>' %t/tu2.ll -o - | \
; RUN:   opt -module-summary -o %t/tu2.bc
; RUN: llvm-lto --thinlto-action=thinlink -o %t/combined %t/tu1.bc %t/tu2.bc
; RUN: llvm-lto --thinlto-action=import  \
; RUN:         --thinlto-index=%t/combined  \
; RUN:         --exported-symbol=main    \
; RUN:         --exported-symbol=f_add   \
; RUN:         --exported-symbol=my_function \
; RUN:        %t/tu2.bc -o %t/tu2.imported.bc
; RUN: llvm-dis %t/tu2.imported.bc -o - | FileCheck %s --check-prefix=CHECK-TU2-IMPORTED

;--- tu1.ll
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

@__loadtime_comment_str_43ac0464497b8531 = weak_odr hidden unnamed_addr constant [14 x i8] c"Copyright TU1\00", align 1, !loadtime_comment !0
@llvm.compiler.used = appending global [1 x ptr] [ptr @__loadtime_comment_str_43ac0464497b8531], section "llvm.metadata"

define i32 @f_add(i32 noundef %a, i32 noundef %b) {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}

!0 = !{}

;--- tu2.ll
target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32-f64:32:64"
target triple = "powerpc-ibm-aix7.3.0.0"

@__loadtime_comment_str_645206960c47d270 = weak_odr hidden unnamed_addr constant [14 x i8] c"Copyright TU2\00", align 1, !loadtime_comment !0
@llvm.compiler.used = appending global [1 x ptr] [ptr @__loadtime_comment_str_645206960c47d270], section "llvm.metadata"

declare i32 @f_add(i32 noundef, i32 noundef)

define i32 @main() {
entry:
  %call = tail call i32 @f_add(i32 noundef 1, i32 noundef 2)
  ret i32 %call
}

!0 = !{}

;; After ThinLTO import, TU2's module should contain both copyright globals.
;; TU2's own string stays weak_odr; TU1's string is imported as available_externally.
; CHECK-TU2-IMPORTED: @[[TU2_STR:__loadtime_comment_str_[0-9a-f]+]] = weak_odr hidden unnamed_addr constant [14 x i8] c"Copyright TU2\00", align 1, !loadtime_comment
; CHECK-TU2-IMPORTED: @[[TU1_STR:__loadtime_comment_str_[0-9a-f]+]] = available_externally hidden unnamed_addr constant [14 x i8] c"Copyright TU1\00", align 1, !loadtime_comment
; CHECK-TU2-IMPORTED: @llvm.compiler.used = appending global [2 x ptr] [ptr @[[TU2_STR]], ptr @[[TU1_STR]]], section "llvm.metadata"

;; main carries TU2's implicit.ref.
; CHECK-TU2-IMPORTED: define i32 @main(){{.*}}!implicit.ref ![[MAIN_MD:[0-9]+]]

;; f_add is imported and carries TU1's implicit.ref.
; CHECK-TU2-IMPORTED: define available_externally {{.*}}i32 @f_add({{.*}}){{.*}}!implicit.ref ![[FADD_MD:[0-9]+]]

;; Verify metadata content.
; CHECK-TU2-IMPORTED: ![[MAIN_MD]] = !{ptr @[[TU2_STR]]}
; CHECK-TU2-IMPORTED: ![[FADD_MD]] = !{ptr @[[TU1_STR]]}
