; RUN: opt -passes=instcombine -S -o - %s | FileCheck %s
; RUN: opt -passes=instcombine -S -o - %s --try-experimental-debuginfo-iterators | FileCheck %s

declare dso_local i32 @bar(ptr)

; Function Attrs: nounwind
define internal i32 @foo() #0 !dbg !1 {
; CHECK:  %[[VLA:.*]] = alloca [2 x i32]
; CHECK:  #dbg_declare(ptr %[[VLA]], {{.*}}, !DIExpression(),

entry:
  %vla = alloca i32, i64 2, align 4, !dbg !16
  call void @llvm.dbg.declare(metadata ptr %vla, metadata !19, metadata !DIExpression()), !dbg !20
  %call = call i32 @bar(ptr %vla), !dbg !22
  unreachable
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram(name: "a", scope: !2, file: !2, line: 232, type: !3, isLocal: true, isDefinition: true, scopeLine: 234, flags: DIFlagPrototyped, isOptimized: true, unit: !5, retainedNodes: !6)
!2 = !DIFile(filename: "b", directory: "c")
!3 = !DISubroutineType(types: !4)
!4 = !{}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4)
!6 = !{!7, !11}
!7 = !DILocalVariable(name: "__vla_expr", scope: !8, type: !10, flags: DIFlagArtificial)
!8 = distinct !DILexicalBlock(scope: !9, file: !2, line: 238, column: 39)
!9 = distinct !DILexicalBlock(scope: !1, file: !2, line: 238, column: 6)
!10 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "ptr32", scope: !8, file: !2, line: 240, type: !12)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, elements: !14)
!13 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!14 = !{!15}
!15 = !DISubrange(count: !7)
!16 = !DILocation(line: 240, column: 3, scope: !17)
!17 = distinct !DILexicalBlock(scope: !18, file: !2, line: 238, column: 39)
!18 = distinct !DILexicalBlock(scope: !1, file: !2, line: 238, column: 6)
!19 = !DILocalVariable(name: "ptr32", scope: !17, file: !2, line: 240, type: !12)
!20 = !DILocation(line: 240, column: 12, scope: !17)
!21 = !DILocation(line: 241, column: 65, scope: !17)
!22 = !DILocation(line: 241, column: 11, scope: !17)
