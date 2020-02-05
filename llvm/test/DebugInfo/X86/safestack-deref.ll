; This test prevents a regression of PR44585 where the DW_OP_deref is incorrectly placed at the end of the expression chain.

; RUN: llc -O0 -mtriple=x86_64-unknown-linux-gnu -filetype=obj %s -o - | llvm-dwarfdump - --name=value | FileCheck %s
; REQUIRES: object-emission

; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_location
; CHECK-NEXT: DW_OP_breg{{[0-9]+}} R{{[0-9A-Z]+}}-8
; CHECK-NEXT: DW_OP_breg{{[0-9]+}} R{{[0-9A-Z]+}}+8, DW_OP_deref, DW_OP_lit8, DW_OP_minus
; CHECK-NEXT: DW_AT_name ("value")

define dso_local void @_Z4funcv() safestack !dbg !7 {
  %1 = alloca i8, align 8
  call void @llvm.dbg.declare(metadata i8* %1, metadata !11, metadata !DIExpression()), !dbg !22
  call void @extern_func(i8* %1), !dbg !22
  ret void
}
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @extern_func(i8* %0)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "/tmp/test3.cpp", directory: "")
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "func", linkageName: "_Z4funcv", scope: !8, file: !8, line: 7, type: !9, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DIFile(filename: "/tmp/test3.cpp", directory: "")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(name: "value", scope: !7, file: !8, line: 8, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !8, line: 1, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !13, identifier: "_ZTS1A")
!13 = !{!14, !18}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !12, file: !8, line: 4, baseType: !15, size: 64, flags: DIFlagPublic)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !17)
!17 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!18 = !DISubprogram(name: "A", scope: !12, file: !8, line: 3, type: !19, scopeLine: 3, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !21, !15}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DILocation(line: 8, column: 5, scope: !7)
