; RUN: llvm-as < %s -disable-output 2>&1 | FileCheck %s

define void @foo(i32 %n) {
entry:
  %0 = zext i32 %n to i64
  %vla = alloca i32, i64 %0, align 16
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !19, metadata !12), !dbg !18
  ret void
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.1", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "vla.c", directory: "/path/to")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 5.0.1"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 20, type: !8, isLocal: false, isDefinition: true, scopeLine: 20, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!16, !19}
!12 = !DIExpression()
!16 = !DILocalVariable(name: "vla_expr", scope: !7, file: !1, line: 21, type: !17)
!17 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!18 = !DILocation(line: 21, column: 7, scope: !7)
!19 = !DILocalVariable(name: "vla", scope: !7, file: !1, line: 21, type: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, align: 32, elements: !21)
!21 = !{!22}
; CHECK: Count must either be a signed constant or a DIVariable
!22 = !DISubrange(count: !17)
