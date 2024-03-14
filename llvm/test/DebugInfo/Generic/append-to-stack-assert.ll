; RUN: opt -passes=instcombine -S < %s | FileCheck %s

@f = dso_local local_unnamed_addr global i16 0, align 1, !dbg !0
@e = dso_local local_unnamed_addr global i159 0, align 1, !dbg !5

; Function Attrs: nounwind
define dso_local void @g() local_unnamed_addr #0 !dbg !14 {
; CHECK:    [[TMP0:%.*]] = load i16, ptr @f, align 1
; CHECK:    [[CONV1:%.*]] = sext i16 [[TMP0]] to i159
; CHECK:    tail call void @llvm.dbg.value(metadata i159 [[CONV1]], metadata {{![0-9]+}}, metadata !DIExpression(DW_OP_LLVM_convert, 159, DW_ATE_signed, DW_OP_LLVM_convert, 256, DW_ATE_signed, DW_OP_stack_value))
entry:
  %0 = load i16, ptr @f, align 1, !dbg !21
  %conv = sext i16 %0 to i256, !dbg !21
  tail call void @llvm.dbg.value(metadata i256 %conv, metadata !18, metadata !DIExpression()), !dbg !22
  %conv1 = trunc i256 %conv to i159, !dbg !23
  store i159 %conv1, ptr @e, align 1, !dbg !23
  ret void, !dbg !23
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "f", scope: !2, file: !3, line: 7, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 19.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "bbi-93380.c", directory: "/")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "e", scope: !2, file: !3, line: 8, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "_BitInt", size: 160, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 1}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang version 19.0.0git"}
!14 = distinct !DISubprogram(name: "g", scope: !3, file: !3, line: 9, type: !15, scopeLine: 9, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18}
!18 = !DILocalVariable(name: "d", scope: !19, file: !3, line: 9, type: !20)
!19 = distinct !DILexicalBlock(scope: !14, file: !3, line: 9)
!20 = !DIBasicType(name: "_BitInt", size: 256, encoding: DW_ATE_signed)
!21 = !DILocation(line: 9, scope: !19)
!22 = !DILocation(line: 0, scope: !19)
!23 = !DILocation(line: 9, scope: !14)
