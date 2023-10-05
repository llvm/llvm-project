; This checks that .debug_names can be generated with monolithic -fdebug-type-sections.

; RUN: llc -mtriple=x86_64 -generate-type-units -dwarf-version=5 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info -debug-names %t | FileCheck %s

; CHECK:     .debug_info contents:
; CHECK:      DW_TAG_type_unit
; CHECK-NEXT:   DW_AT_language  (DW_LANG_C_plus_plus_14)
; CHECK-NEXT:   DW_AT_stmt_list (0x00000000)
; CHECK-NEXT:   DW_AT_str_offsets_base  (0x00000008)
; CHECK:      DW_TAG_structure_type
; CHECK-NEXT:     DW_AT_calling_convention  (DW_CC_pass_by_value)
; CHECK-NEXT:     DW_AT_name  ("Foo")
; CHECK-NEXT:     DW_AT_byte_size (0x08)
; CHECK-NEXT:     DW_AT_decl_file ("/typeSmall/main.cpp")
; CHECK-NEXT:     DW_AT_decl_line (1)
; CHECK:       DW_TAG_member
; CHECK-NEXT:       DW_AT_name  ("c1")
; CHECK-NEXT:       DW_AT_type  (0x00000033 "char *")
; CHECK-NEXT:       DW_AT_decl_file ("/typeSmall/main.cpp")
; CHECK-NEXT:       DW_AT_decl_line (2)
; CHECK-NEXT:       DW_AT_data_member_location  (0x00)
; CHECK:       DW_TAG_pointer_type
; CHECK-NEXT:     DW_AT_type  (0x00000038 "char")
; CHECK:       DW_TAG_base_type
; CHECK-NEXT:     DW_AT_name  ("char")
; CHECK-NEXT:     DW_AT_encoding  (DW_ATE_signed_char)
; CHECK-NEXT:     DW_AT_byte_size (0x01)
; CHECK:        .debug_names contents:
; CHECK:         Compilation Unit offsets [
; CHECK-NEXT:        CU[0]: 0x00000000
; CHECK-NEXT:      ]
; CHECK-NEXT:      Local Type Unit offsets [
; CHECK-NEXT:        LocalTU[0]: 0x00000000
; CHECK-NEXT:      ]
; CHECK:        Abbreviations [
; CHECK-NEXT:     Abbreviation [[ABBREV1:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_structure_type
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:     }
; CHECK-NEXT:     Abbreviation [[ABBREV3:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_structure_type
; CHECK-NEXT:       DW_IDX_type_unit: DW_FORM_data1
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:     }
; CHECK-NEXT:     Abbreviation [[ABBREV:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_base_type
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:     }
; CHECK-NEXT:     Abbreviation [[ABBREV2:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_subprogram
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:     }
; CHECK-NEXT:     Abbreviation [[ABBREV4:0x[0-9a-f]*]] {
; CHECK-NEXT:       Tag: DW_TAG_base_type
; CHECK-NEXT:       DW_IDX_type_unit: DW_FORM_data1
; CHECK-NEXT:       DW_IDX_die_offset: DW_FORM_ref4
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 0 [
; CHECK-NEXT:     Name 1 {
; CHECK-NEXT:       Hash: 0xB888030
; CHECK-NEXT:       String: {{.+}} "int"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: [[ABBREV]] {
; CHECK-NEXT:           Tag: DW_TAG_base_type
; CHECK-NEXT:           DW_IDX_die_offset: 0x0000003e
; CHECK-NEXT:         }
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 1 [
; CHECK-NEXT:     Name 2 {
; CHECK-NEXT:       Hash: 0xB887389
; CHECK-NEXT:       String: {{.+}} "Foo"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: [[ABBREV3]] {
; CHECK-NEXT:           Tag: DW_TAG_structure_type
; CHECK-NEXT:           DW_IDX_type_unit: 0x00
; CHECK-NEXT:           DW_IDX_die_offset: 0x00000023
; CHECK-NEXT:         }
; CHECK-NEXT:       }
; CHECK-NEXT:       Entry @ 0xaa {
; CHECK-NEXT:         Abbrev: [[ABBREV1]] {
; CHECK-NEXT:           Tag: DW_TAG_structure_type
; CHECK-NEXT:           DW_IDX_die_offset: 0x00000042
; CHECK-NEXT:         }
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 2 [
; CHECK-NEXT:     Name 3 {
; CHECK-NEXT:       Hash: 0x7C9A7F6A
; CHECK-NEXT:       String: {{.+}} "main"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: [[ABBREV2]] {
; CHECK-NEXT:           Tag: DW_TAG_subprogram
; CHECK-NEXT:           DW_IDX_die_offset: 0x00000023
; CHECK-NEXT:         }
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT:   Bucket 3 [
; CHECK-NEXT:     Name 4 {
; CHECK-NEXT:       Hash: 0x7C952063
; CHECK-NEXT:       String: {{.+}} "char"
; CHECK-NEXT:       Entry @ {{.+}} {
; CHECK-NEXT:         Abbrev: [[ABBREV4]] {
; CHECK-NEXT:           Tag: DW_TAG_base_type
; CHECK-NEXT:           DW_IDX_type_unit: 0x00
; CHECK-NEXT:           DW_IDX_die_offset: 0x00000038
; CHECK-NEXT:         }
; CHECK-NEXT:       }
; CHECK-NEXT:     }
; CHECK-NEXT:   ]
; CHECK-NEXT: }


; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Foo = type { ptr }

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #0 !dbg !10 {
entry:
  %retval = alloca i32, align 4
  %f = alloca %struct.Foo, align 8
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %f, metadata !15, metadata !DIExpression()), !dbg !21
  ret i32 0, !dbg !22
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { mustprogress noinline norecurse nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!1 = !DIFile(filename: "main.cpp", directory: "/typeSmall", checksumkind: CSK_MD5, checksum: "e5b402e9dbafe24c7adbb087d1f03549")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 18.0.0"}
!10 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "f", scope: !10, file: !1, line: 5, type: !16)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !17, identifier: "_ZTS3Foo")
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "c1", scope: !16, file: !1, line: 2, baseType: !19, size: 64)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!21 = !DILocation(line: 5, column: 6, scope: !10)
!22 = !DILocation(line: 6, column: 2, scope: !10)
