; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O2 -regalloc=basic < %s | FileCheck %s
; Test to check that unused argument 'this' is not undefined in debug info.

target triple = "x86_64-apple-darwin10.2"
%struct.foo = type { i32 }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (%struct.foo*, i32)* @_ZN3foo3bazEi to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i32 @_ZN3foo3bazEi(%struct.foo* nocapture %this, i32 %x) nounwind readnone optsize noinline ssp align 2 !dbg !8 {
;CHECK: DEBUG_VALUE: baz:this <- %RDI{{$}}
entry:
  tail call void @llvm.dbg.value(metadata %struct.foo* %this, i64 0, metadata !15, metadata !DIExpression()), !dbg !DILocation(scope: !8)
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !16, metadata !DIExpression()), !dbg !DILocation(scope: !8)
  %0 = mul nsw i32 %x, 7, !dbg !29                ; <i32> [#uses=1]
  %1 = add nsw i32 %0, 1, !dbg !29                ; <i32> [#uses=1]
  ret i32 %1, !dbg !29
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!34}
!llvm.dbg.lv = !{!0, !14, !15, !16, !17, !24, !25, !28}

!0 = !DILocalVariable(name: "this", line: 11, arg: 1, scope: !1, file: !3, type: !12)
!1 = distinct !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEi", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !4, scopeLine: 11, file: !31, scope: !2, type: !9)
!2 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", line: 3, size: 32, align: 32, file: !31, scope: !3, elements: !5)
!3 = !DIFile(filename: "foo.cp", directory: "/tmp/")
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "4.2.1 LLVM build", isOptimized: true, emissionKind: FullDebug, file: !31, enums: !32, retainedTypes: !32)
!5 = !{!6, !1, !8}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "y", line: 8, size: 32, align: 32, file: !31, scope: !2, baseType: !7)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "baz", linkageName: "_ZN3foo3bazEi", line: 15, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !4, scopeLine: 15, file: !31, scope: !2, type: !9)
!9 = !DISubroutineType(types: !10)
!10 = !{!7, !11, !7}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial, file: !31, scope: !3, baseType: !2)
!12 = !DIDerivedType(tag: DW_TAG_const_type, size: 64, align: 64, flags: DIFlagArtificial, file: !31, scope: !3, baseType: !13)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !31, scope: !3, baseType: !2)
!14 = !DILocalVariable(name: "x", line: 11, arg: 2, scope: !1, file: !3, type: !7)
!15 = !DILocalVariable(name: "this", line: 15, arg: 1, scope: !8, file: !3, type: !12)
!16 = !DILocalVariable(name: "x", line: 15, arg: 2, scope: !8, file: !3, type: !7)
!17 = !DILocalVariable(name: "argc", line: 19, arg: 1, scope: !18, file: !3, type: !7)
!18 = distinct !DISubprogram(name: "main", linkageName: "main", line: 19, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !4, scopeLine: 19, file: !31, scope: !3, type: !19)
!19 = !DISubroutineType(types: !20)
!20 = !{!7, !7, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !31, scope: !3, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !31, scope: !3, baseType: !23)
!23 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!24 = !DILocalVariable(name: "argv", line: 19, arg: 2, scope: !18, file: !3, type: !21)
!25 = !DILocalVariable(name: "a", line: 20, scope: !26, file: !3, type: !2)
!26 = distinct !DILexicalBlock(line: 19, column: 0, file: !31, scope: !27)
!27 = distinct !DILexicalBlock(line: 19, column: 0, file: !31, scope: !18)
!28 = !DILocalVariable(name: "b", line: 21, scope: !26, file: !3, type: !7)
!29 = !DILocation(line: 16, scope: !30)
!30 = distinct !DILexicalBlock(line: 15, column: 0, file: !31, scope: !8)
!31 = !DIFile(filename: "foo.cp", directory: "/tmp/")
!32 = !{}
!34 = !{i32 1, !"Debug Info Version", i32 3}
