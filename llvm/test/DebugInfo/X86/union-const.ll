; RUN: llc -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]	(0)
; CHECK-NEXT: DW_AT_name {{.*}}"a"
;
; ModuleID = 'union.c'
; generated at -O1 from:
; union mfi_evt {
;   struct {
;     int reserved;
;   } members;
; } mfi_aen_setup() {
;   union mfi_evt a;
;   a.members.reserved = 0;
; }
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%union.mfi_evt = type { %struct.anon }
%struct.anon = type { i32 }

; Function Attrs: nounwind readnone ssp uwtable
define i32 @mfi_aen_setup() #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.declare(metadata %union.mfi_evt* undef, metadata !16, metadata !21), !dbg !22
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !16, metadata !21), !dbg !22
  ret i32 undef, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind readnone ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.7.0 (trunk 226915) (llvm/trunk 226905)", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "union.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "mfi_aen_setup", line: 5, isLocal: false, isDefinition: true, isOptimized: true, scopeLine: 5, file: !1, scope: !5, type: !6, variables: !15)
!5 = !DIFile(filename: "union.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DICompositeType(tag: DW_TAG_union_type, name: "mfi_evt", line: 1, size: 32, align: 32, file: !1, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "members", line: 4, size: 32, align: 32, file: !1, scope: !8, baseType: !11)
!11 = !DICompositeType(tag: DW_TAG_structure_type, line: 2, size: 32, align: 32, file: !1, scope: !8, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "reserved", line: 3, size: 32, align: 32, file: !1, scope: !11, baseType: !14)
!14 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DILocalVariable(name: "a", line: 6, scope: !4, file: !5, type: !8)
!17 = !{i32 2, !"Dwarf Version", i32 2}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"PIC Level", i32 2}
!20 = !{!"clang version 3.7.0 (trunk 226915) (llvm/trunk 226905)"}
!21 = !DIExpression()
!22 = !DILocation(line: 6, column: 17, scope: !4)
!23 = !DILocation(line: 8, column: 1, scope: !4)
