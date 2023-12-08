; RUN: llc -mtriple mips-linux-gnu -O2 %s -o - | FileCheck %s
; Check struct X for dead variable xyz from inlined function foo.

; CHECK: .section .debug_info,"",@0x7000001e
; CHECK:	DW_TAG_structure_type
; CHECK-NEXT:	info_string

source_filename = "test/DebugInfo/Mips/InlinedFnLocalVar.ll"

@i = common global i32 0, !dbg !0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; Function Attrs: nounwind ssp
define i32 @bar() #1 !dbg !8 {
entry:
  %0 = load i32, ptr @i, align 4, !dbg !11
  tail call void @llvm.dbg.value(metadata i32 %0, metadata !13, metadata !24), !dbg !25
  tail call void @llvm.dbg.declare(metadata !5, metadata !18, metadata !24), !dbg !26
  %1 = mul nsw i32 %0, %0, !dbg !27
  store i32 %1, ptr @i, align 4, !dbg !11
  ret i32 %1, !dbg !28
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind ssp }

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "i", scope: !2, file: !2, line: 5, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "bar.c", directory: "/tmp/")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C89, file: !2, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5)
!5 = !{}
!6 = !{!0}
!7 = !{i32 1, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "bar", linkageName: "bar", scope: !2, file: !2, line: 14, type: !9, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !4)
!9 = !DISubroutineType(types: !10)
!10 = !{!3}
!11 = !DILocation(line: 15, scope: !12)
!12 = distinct !DILexicalBlock(scope: !8, file: !2, line: 14)
!13 = !DILocalVariable(name: "j", arg: 1, scope: !14, file: !2, line: 9, type: !3)
!14 = distinct !DISubprogram(name: "foo", scope: !2, file: !2, line: 9, type: !15, isLocal: true, isDefinition: true, scopeLine: 9, virtualIndex: 6, isOptimized: true, unit: !4, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{!3, !3}
!17 = !{!13, !18}
!18 = !DILocalVariable(name: "xyz", scope: !19, file: !2, line: 10, type: !20)
!19 = distinct !DILexicalBlock(scope: !14, file: !2, line: 9)
!20 = !DICompositeType(tag: DW_TAG_structure_type, name: "X", scope: !14, file: !2, line: 10, size: 64, align: 32, elements: !21)
!21 = !{!22, !23}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !20, file: !2, line: 10, baseType: !3, size: 32, align: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !20, file: !2, line: 10, baseType: !3, size: 32, align: 32, offset: 32)
!24 = !DIExpression()
!25 = !DILocation(line: 9, scope: !14, inlinedAt: !11)
!26 = !DILocation(line: 9, scope: !19, inlinedAt: !11)
!27 = !DILocation(line: 11, scope: !19, inlinedAt: !11)
!28 = !DILocation(line: 16, scope: !12)

