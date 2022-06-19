; RUN: llc < %s -mtriple=s390x-linux-gnu -generate-arange-section
;
; Test that the EXRL target instruction is emitted successfully (before text
; section is closed).

@a = dso_local global i32* null, align 8, !dbg !0
@j = dso_local global i32 0, align 4, !dbg !5

define void @fun() !dbg !14 {
entry:
  %0 = load i32*, i32** @a, align 8, !dbg !18
  %1 = bitcast i32* %0 to i8*, !dbg !19
  %2 = load i32, i32* @j, align 4, !dbg !20
  %conv = sext i32 %2 to i64, !dbg !20
  call void @llvm.memset.p0i8.i64(i8* align 4 %1, i8 0, i64 %conv, i1 false), !dbg !19
  ret void, !dbg !21
}

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #1

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "file.c", directory: "")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "j", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang version 14.0.0"}
!14 = distinct !DISubprogram(name: "fun", scope: !3, file: !3, line: 3, type: !15, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{}
!18 = !DILocation(line: 4, column: 11, scope: !14)
!19 = !DILocation(line: 4, column: 4, scope: !14)
!20 = !DILocation(line: 4, column: 15, scope: !14)
!21 = !DILocation(line: 5, column: 1, scope: !14)
