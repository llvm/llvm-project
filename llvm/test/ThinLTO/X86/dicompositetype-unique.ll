; RUN: opt -module-summary -o %t1.bc %s
; RUN: opt -module-summary -o %t2.bc %S/Inputs/dicompositetype-unique.ll

; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t --save-temps \
; RUN:    -r %t1.bc,_foo,lx \
; RUN:    -r %t1.bc,_main,plx \
; RUN:    -r %t2.bc,_foo,plx
; RUN: llvm-dis %t.1.3.import.bc -o - | FileCheck %s


target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare void @foo()
define i32 @main() {
  call void @foo()
  ret i32 0
}


; Check that the type map will unique two DICompositeTypes.
; CHECK:        !DICompositeType(
; CHECK-SAME:        name: "T1"
; CHECK-SAME:        identifier: "T"
; CHECK-NOT:    T2

!named = !{!0, !1, !2, !3}
!0 = !DIFile(filename: "abc", directory: "/path/to")
!1 = !DICompositeType(tag: DW_TAG_class_type, name: "T1", identifier: "T", file: !0)
!2 = !DICompositeType(tag: DW_TAG_class_type, flags: DIFlagFwdDecl, name: "FwdTFwd", identifier: "FwdT", file: !0)
!3 = !DICompositeType(tag: DW_TAG_class_type, flags: DIFlagFwdDecl, name: "BothFwdT1", identifier: "BothFwdT", file: !0)
