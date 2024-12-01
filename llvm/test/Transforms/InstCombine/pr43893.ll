; Check for setting dbg.value as undef which depends on trivially dead instructions.
; RUN: opt -passes=instcombine -S -o - %s | FileCheck %s
; RUN: opt -passes=instcombine -S -o - %s --try-experimental-debuginfo-iterators | FileCheck %s

@a = common dso_local global i8 0, align 1, !dbg !0
@b = common dso_local global i8 0, align 1, !dbg !6

define dso_local i32 @main() !dbg !13 {
entry:
  %0 = load i8, ptr @a, align 1, !dbg !17
  %dec = add i8 %0, -1, !dbg !17
  store i8 %dec, ptr @a, align 1, !dbg !17
;CHECK: #dbg_value(i32 poison
;CHECK: #dbg_value(i32 -8
;CHECK: #dbg_value(i32 poison
  %conv = sext i8 %dec to i32, !dbg !17
  %udiv = udiv i32 %conv, 4, !dbg !17
  call void @llvm.dbg.value(metadata i32 %udiv, metadata !18, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 -8, metadata !20, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i32 %udiv, metadata !20, metadata !DIExpression()), !dbg !19
  store i8 0, ptr @b, align 1, !dbg !21
  %cmp = icmp sgt i32 %conv, 0, !dbg !22
  %conv1 = zext i1 %cmp to i32, !dbg !22
  ret i32 0, !dbg !23
}

declare void @llvm.dbg.value(metadata, metadata, metadata) 

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test", directory: "n")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 10.0.0"}
!13 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 2, type: !14, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!14 = !DISubroutineType(types: !15)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 4, column: 11, scope: !13)
!18 = !DILocalVariable(name: "c", scope: !13, file: !3, line: 4, type: !16)
!19 = !DILocation(line: 0, scope: !13)
!20 = !DILocalVariable(name: "l_1240", scope: !13, file: !3, line: 6, type: !16)
!21 = !DILocation(line: 10, column: 10, scope: !13)
!22 = !DILocation(line: 10, column: 5, scope: !13)
!23 = !DILocation(line: 12, column: 1, scope: !13)
