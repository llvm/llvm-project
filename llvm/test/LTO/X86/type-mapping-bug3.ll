; RUN: opt -module-summary -o %t0.o %S/Inputs/type-mapping-bug3.ll
; RUN: opt -module-summary -o %t1.o %s
; RUN: llvm-lto2 run -o %t2 %t0.o %t1.o -r %t0.o,a,px -r %t1.o,b,px -r %t1.o,c,px -r %t1.o,d,
;
; Test for the issue described in https://bugs.llvm.org/show_bug.cgi?id=40312

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; T2 is the non-opaque struct required to trigger the uniqued T2.0 and T3.0 to
; respectively T2 and T3 in the destination module.
%"T2" = type { %"T3"* }
%"T3" = type opaque

; Use/refer to T2 so it gets added as an IdentifiedStructType.
define void @c(%"T2") {
    unreachable
}

; The global declaration that causes the assertion when its type is mapped to
; itself incorrectly.
declare void @d(%"T3"*)

define void @b() {
entry:
  %f.addr = alloca %"T3"*load %"T3"*, %"T3"** %f.addr

  ; The call with the getCalledValue() vs getCalledFunction() mismatch.
  call void @d(%"T3"* %0)
  unreachable
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, retainedTypes: !4)
!3 = !DIFile(filename: "f1", directory: "")
!4 = !{!5}

; This DICompositeType is referenced by !5 in Inputs/type-mapping-bug3.ll
; causing the function type in !7 to be added to its module.
!5 = !DICompositeType(tag: DW_TAG_structure_type, templateParams: !6, identifier: "SHARED")
!6 = !{!7}

; The reference to d and T3 that gets loaded into %t0.o
!7 = !DITemplateValueParameter(value: void (%"T3"*)* @d)
