; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc -bitcode-mdindex-threshold=0
; RUN: opt -module-summary %p/Inputs/lazyload_metadata.ll -o %t2.bc -bitcode-mdindex-threshold=0
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %t2.bc
; REQUIRES: asserts

; Check that importing @globalfunc1 does not trigger loading all the global
; metadata for @globalfunc2 and @globalfunc3

; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t3.bc -stats 2> %t4.txt
; RUN: llvm-lto -thinlto-action=import %t2.bc -thinlto-index=%t3.bc -stats -disable-ondemand-mds-loading 2> %t5.txt
; RUN: cat %t4.txt %t5.txt | FileCheck %s

; Check llvm-lto call with lazy loading enabled
; CHECK: [[#LAZY_RECORDS:]] bitcode-reader  - Number of Metadata records loaded
; CHECK: 2 bitcode-reader  - Number of MDStrings loaded

; Check llvm-lto call with lazy loading disabled
; CHECK: [[#LAZY_RECORDS+9]] bitcode-reader  - Number of Metadata records loaded
; CHECK: 7 bitcode-reader  - Number of MDStrings loaded

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @globalfunc1(i32 %arg) {
  %x = call i1 @llvm.type.test(ptr undef, metadata !"typeid1")
  %tmp = add i32 %arg, 0, !metadata !2
  ret void
}

; We need two functions here that will both reference the same metadata.
; This is to force the metadata to be emitted in the global metadata block and
; not in the function specific metadata.
; These function are not imported and so we don't want to load their metadata.

define void @globalfunc2(i32 %arg) {
  %x = call i1 @llvm.type.test(ptr undef, metadata !"typeid1")
  %tmp = add i32 %arg, 0, !metadata !1
  ret void
}

define void @globalfunc3(i32 %arg) {
  %tmp = add i32 %arg, 0, !metadata !1
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"Hello World"}
!3 = !{!"3"}
!4 = !{!"4"}
!5 = !{!"5"}
!6 = !{!9}
!7 = !{!"7"}
!8 = !{!"8"}
!9 = !{!6}
