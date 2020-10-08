; RUN: opt -module-summary -o %t0.o %S/Inputs/type-mapping-bug4_0.ll
; RUN: opt -module-summary -o %t1.o %S/Inputs/type-mapping-bug4_1.ll
; RUN: opt -module-summary -o %t2.o %s
; RUN: llvm-lto2 run -save-temps -o %t3 %t0.o %t1.o %t2.o -r %t1.o,a,px -r %t2.o,d,px -r %t1.o,h,x -r %t2.o,h,x -r %t1.o,j,px
; RUN: llvm-dis < %t3.0.0.preopt.bc | FileCheck %s

; stage0: linking t0.o
; stage1: linking t1.o
; stage2: during linking t1.o, mapping @d
; stage3: linking t2.o

; Stage0 is not described because it is not interesting for the purpose of this test.
; Stage1 and stage2 are described in type-mapping-bug4_1.ll.
; Stage3 is described in this file.

; CHECK: %class.CCSM = type opaque
; CHECK: %class.CB = type { %"class.std::unique_ptr_base.1" }
; CHECK: %"class.std::unique_ptr_base.1" = type { %class.CCSM* }

; CHECK: define void @j() {
; CHECK:   call void @h(%class.CCSM* undef)
; CHECK:   ret void
; CHECK: }

; CHECK: declare void @h(%class.CCSM*)

; CHECK: define void @a() {
; CHECK:   call void @llvm.dbg.value(metadata %class.CB* undef, metadata !10, metadata !DIExpression())
; CHECK:   ret void
; CHECK: }

; CHECK: declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; CHECK: define void @d(%class.CB* %0) {
; CHECK:   %2 = getelementptr inbounds %class.CB, %class.CB* undef, i64 0, i32 0, i32 0
; CHECK:   ret void
; CHECK: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; (stage3) Remapping this type returns itself due to D47898 and stage1.3
%class.CB = type { %"class.std::unique_ptr_base.2" }

; (stage3) Remapping this type returns itself due to D47898 and stage2
%"class.std::unique_ptr_base.2" = type { %class.CCSM* }

%class.CCSM = type opaque

; (stage3) computeTypeMapping add the mapping %class.CCSM -> %class.CWBD due to stage1.2
declare void @h(%class.CCSM*)

define void @d(%class.CB*) {
  ; Without the fix in D87001 to delay materialization of @d until its module is linked
  ; (stage3)
  ; * SourceElementType of getelementptr is remapped to itself.
  ; * ResultElementType of getelementptr is incorrectly remapped to %class.CWBD*.
  ;   Its type should be %class.CCSM*.
  %2 = getelementptr inbounds %class.CB, %class.CB* undef, i64 0, i32 0, i32 0
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"ThinLTO", i32 0}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, retainedTypes: !4)
!3 = !DIFile(filename: "f1", directory: "")
!4 = !{!5}

; This DICompositeType is referenced by !5 in Inputs/type-mapping-bug4_1.ll
; causing the function type in !7 to be added to its module.
!5 = !DICompositeType(tag: DW_TAG_structure_type, templateParams: !6, identifier: "SHARED")
!6 = !{!7}

; The reference to d and %class.CB that gets loaded into %t1.o
!7 = !DITemplateValueParameter(value: void (%class.CB*)* @d)
