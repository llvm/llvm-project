; RUN: llvm-dis -o - %s.bc | FileCheck %s
%class.A = type { i32, i32, i32, i32 }

define void @_Z3fooi(%class.A* sret %agg.result) #0 !dbg !3 {
  ; CHECK: call void @llvm.dbg.declare({{.*}}, metadata ![[EXPR:[0-9]+]]), !dbg
  ; CHECK: ![[EXPR]] = !DIExpression()
  call void @llvm.dbg.declare(metadata %class.A* %agg.result, metadata !13, metadata !16), !dbg !17
  ret void, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { ssp }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.cc", directory: "/tmp")
!2 = !{i32 1, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 4, type: !4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DICompositeType(tag: DW_TAG_class_type, name: "A", scope: !0, file: !1, line: 2, size: 128, align: 32, elements: !7)
!7 = !{!8, !10, !11, !12}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1, file: !1, line: 2, baseType: !9, size: 32, align: 32)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1, file: !1, line: 2, baseType: !9, size: 32, align: 32, offset: 32)
!11 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1, file: !1, line: 2, baseType: !9, size: 32, align: 32, offset: 64)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "o", scope: !1, file: !1, line: 2, baseType: !9, size: 32, align: 32, offset: 96)
!13 = !DILocalVariable(name: "my_a", scope: !14, file: !1, line: 9, type: !15)
!14 = distinct !DILexicalBlock(scope: !3, file: !1, line: 4, column: 14)
!15 = !DIDerivedType(tag: DW_TAG_reference_type, file: !1, baseType: !6)
!16 = !DIExpression(DW_OP_deref)
!17 = !DILocation(line: 9, column: 5, scope: !3)
