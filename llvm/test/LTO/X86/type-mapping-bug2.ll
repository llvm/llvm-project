; RUN: opt -module-summary -o %t0.o %S/Inputs/type-mapping-bug2.ll
; RUN: opt -module-summary -o %t1.o %s
; RUN: llvm-lto2 run -o %t2 %t0.o %t1.o -r %t0.o,c,px -r %t1.o,a,px -r %t1.o,b,px
;
; Test for the issue described in https://bugs.llvm.org/show_bug.cgi?id=37684

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; T1 will be linked against T2 because T2 was already loaded in %t0.o due to
; the declaration for @b being imported due to !13
%"T1" = type {}
%"T2" = type {}

define %"T1" @a() {
  unreachable
}

define i1 @b(%"T2"*) {
  unreachable
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"ThinLTO", i32 0}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, retainedTypes: !4)
!3 = !DIFile(filename: "f1", directory: "")
!4 = !{!5, !9}
!5 = !DICompositeType(tag: DW_TAG_class_type, file: !3, templateParams: !6, scope: !8)
!6 = !{!7}

; The reference to @b and T2 that will be loaded in %t0.o

!7 = !DITemplateValueParameter(value: i1 (%"T2"*)* @b)
!8 = distinct !DISubprogram(unit: !2)

; This DICompositeType is uniqued against !5 in Inputs/type-mapping-bug2.ll,
; causing !7 and hence %T2 to be loaded into it's module

!9 = !DICompositeType(tag: DW_TAG_array_type, identifier: "SHARED", scope: !8)

