; REQUIRES: object-emission
; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Manually derived by externalizing the composite types from:
;
;   namespace N { class B; }
;   using N::B;
;   class A;
;   A *a;
;
; Test the direct use of an external type.
; CHECK: DW_TAG_variable
; CHECK:   DW_AT_type [DW_FORM_ref4]	  {{.*}}{[[PTR:.*]]}
; CHECK: [[PTR]]: DW_TAG_pointer_type
; CHECK:   DW_AT_type [DW_FORM_ref4]  	  {{.*}}{[[A:.*]]}
; CHECK: [[A]]: DW_TAG_class_type
; CHECK:   DW_AT_declaration [DW_FORM_flag]	(0x01)
; CHECK:   DW_AT_signature [DW_FORM_ref_sig8]	(0x4e834ea939695c24)
; CHECK: [[B:.*]]: DW_TAG_class_type
; CHECK:   DW_AT_declaration [DW_FORM_flag]	(0x01)
; CHECK:   DW_AT_signature [DW_FORM_ref_sig8]	(0x942e51c7addda5f7)
; CHECK:   DW_TAG_imported_declaration
; CHECK:     DW_AT_import [DW_FORM_ref4]  {{.*}}[[B]]

source_filename = "test/DebugInfo/X86/externaltyperef.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%class.A = type opaque

@a = global %class.A* null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 2, type: !11, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.7.0 (trunk 242039) (llvm/trunk 242046)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !8, imports: !9)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{}
!5 = !{!6, !7}
!6 = !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !3, flags: DIFlagExternalTypeRef, identifier: "_ZTS1A")
!7 = !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !3, flags: DIFlagExternalTypeRef, identifier: "_ZTSN1N1BE")
!8 = !{!0}
!9 = !{!10}
!10 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !7, line: 4)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, align: 64)
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"PIC Level", i32 2}
!15 = !{!"clang version 3.7.0 (trunk 242039) (llvm/trunk 242046)"}

