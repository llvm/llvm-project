; RUN: llc -march=x86 -O0 < %s | FileCheck %s
; Currently, dbg.declare generates a DEBUG_VALUE comment.  Eventually it will
; generate DWARF and this test will need to be modified or removed.


%struct.Pt = type { double, double }
%struct.Rect = type { %struct.Pt, %struct.Pt }

define double @foo(%struct.Rect* byval %my_r0) nounwind ssp !dbg !1 {
entry:
;CHECK: DEBUG_VALUE
  %retval = alloca double                         ; <double*> [#uses=2]
  %0 = alloca double                              ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata %struct.Rect* %my_r0, metadata !0, metadata !DIExpression()), !dbg !15
  %1 = getelementptr inbounds %struct.Rect, %struct.Rect* %my_r0, i32 0, i32 0, !dbg !16 ; <%struct.Pt*> [#uses=1]
  %2 = getelementptr inbounds %struct.Pt, %struct.Pt* %1, i32 0, i32 0, !dbg !16 ; <double*> [#uses=1]
  %3 = load double, double* %2, align 8, !dbg !16         ; <double> [#uses=1]
  store double %3, double* %0, align 8, !dbg !16
  %4 = load double, double* %0, align 8, !dbg !16         ; <double> [#uses=1]
  store double %4, double* %retval, align 8, !dbg !16
  br label %return, !dbg !16

return:                                           ; preds = %entry
  %retval1 = load double, double* %retval, !dbg !16       ; <double> [#uses=1]
  ret double %retval1, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!21}

!0 = !DILocalVariable(name: "my_r0", line: 11, arg: 1, scope: !1, file: !2, type: !7)
!1 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, scopeLine: 11, file: !19, scope: !2, type: !4)
!2 = !DIFile(filename: "b2.c", directory: "/tmp/")
!3 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: false, emissionKind: FullDebug, file: !19, enums: !20, retainedTypes: !20, subprograms: !18)
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !7}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "Rect", line: 6, size: 256, align: 64, file: !19, scope: !2, elements: !8)
!8 = !{!9, !14}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "P1", line: 7, size: 128, align: 64, file: !19, scope: !7, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "Pt", line: 1, size: 128, align: 64, file: !19, scope: !2, elements: !11)
!11 = !{!12, !13}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 2, size: 64, align: 64, file: !19, scope: !10, baseType: !6)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "y", line: 3, size: 64, align: 64, offset: 64, file: !19, scope: !10, baseType: !6)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "P2", line: 8, size: 128, align: 64, offset: 128, file: !19, scope: !7, baseType: !10)
!15 = !DILocation(line: 11, scope: !1)
!16 = !DILocation(line: 12, scope: !17)
!17 = distinct !DILexicalBlock(line: 11, column: 0, file: !19, scope: !1)
!18 = !{!1}
!19 = !DIFile(filename: "b2.c", directory: "/tmp/")
!20 = !{}
!21 = !{i32 1, !"Debug Info Version", i32 3}
