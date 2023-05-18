; We generate invalid TBAA, hence -disable-verify, but this is a convenient way
; to trigger a metadata lazyloading crash

; RUN: opt -module-summary %s -o %t.bc -bitcode-mdindex-threshold=0 -disable-verify
; RUN: opt -module-summary %p/Inputs/funcimport-tbaa.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc


; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t3.bc -o - \
; RUN:  | llvm-dis -o - | FileCheck %s --check-prefix=IMPORTGLOB1
; IMPORTGLOB1: define available_externally float @globalfunc1

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define float @globalfunc1(ptr, ptr) {
  %3 = load i32, ptr %0, align 4, !tbaa !0
  %4 = sitofp i32 %3 to float
  %5 = load float, ptr %1, align 4, !tbaa !4
  %6 = fadd float %4, %5
  ret float %6
}

; We need a second function for force the metadata to be emitted in the global block
define float @globalfunc2(ptr, ptr) {
  %3 = load i32, ptr %0, align 4, !tbaa !0
  %4 = sitofp i32 %3 to float
  %5 = load float, ptr %1, align 4, !tbaa !4
  %6 = fadd float %4, %5
  ret float %6
}

!0 = !{!1, !4, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !2, i64 0}
