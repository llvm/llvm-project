; ModuleID = 'pr-43860.cpp'
source_filename = "pr-43860.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z4testii(i32 noundef %Param_1, i32 noundef %Param_2) #0 !dbg !10 {
entry:
  %Param.addr.i = alloca i32, align 4
  %Var_1.i = alloca i32, align 4
  %Var_2.i = alloca i32, align 4
  %Param_1.addr = alloca i32, align 4
  %Param_2.addr = alloca i32, align 4
  %A = alloca i32, align 4
  store i32 %Param_1, ptr %Param_1.addr, align 4
    #dbg_declare(ptr %Param_1.addr, !15, !DIExpression(), !16)
  store i32 %Param_2, ptr %Param_2.addr, align 4
    #dbg_declare(ptr %Param_2.addr, !17, !DIExpression(), !18)
    #dbg_declare(ptr %A, !19, !DIExpression(), !20)
  %0 = load i32, ptr %Param_1.addr, align 4, !dbg !21
  store i32 %0, ptr %A, align 4, !dbg !20
  %1 = load i32, ptr %Param_2.addr, align 4, !dbg !22
  store i32 %1, ptr %Param.addr.i, align 4
    #dbg_declare(ptr %Param.addr.i, !23, !DIExpression(), !27)
    #dbg_declare(ptr %Var_1.i, !29, !DIExpression(), !30)
  %2 = load i32, ptr %Param.addr.i, align 4, !dbg !31
  store i32 %2, ptr %Var_1.i, align 4, !dbg !30
    #dbg_declare(ptr %Var_2.i, !32, !DIExpression(), !34)
  %3 = load i32, ptr %Param.addr.i, align 4, !dbg !35
  %4 = load i32, ptr %Var_1.i, align 4, !dbg !36
  %add.i = add nsw i32 %3, %4, !dbg !37
  store i32 %add.i, ptr %Var_2.i, align 4, !dbg !34
  %5 = load i32, ptr %Var_2.i, align 4, !dbg !38
  store i32 %5, ptr %Var_1.i, align 4, !dbg !39
  %6 = load i32, ptr %Var_1.i, align 4, !dbg !40
  %7 = load i32, ptr %A, align 4, !dbg !41
  %add = add nsw i32 %7, %6, !dbg !41
  store i32 %add, ptr %A, align 4, !dbg !41
  %8 = load i32, ptr %A, align 4, !dbg !42
  ret i32 %8, !dbg !43
}

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "pr-43860.cpp", directory: "/data/projects/scripts/regression-suite/input/general")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 20.0.0"}
!10 = distinct !DISubprogram(name: "test", linkageName: "_Z4testii", scope: !1, file: !1, line: 11, type: !11, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "Param_1", arg: 1, scope: !10, file: !1, line: 11, type: !13)
!16 = !DILocation(line: 11, column: 14, scope: !10)
!17 = !DILocalVariable(name: "Param_2", arg: 2, scope: !10, file: !1, line: 11, type: !13)
!18 = !DILocation(line: 11, column: 27, scope: !10)
!19 = !DILocalVariable(name: "A", scope: !10, file: !1, line: 12, type: !13)
!20 = !DILocation(line: 12, column: 7, scope: !10)
!21 = !DILocation(line: 12, column: 11, scope: !10)
!22 = !DILocation(line: 13, column: 23, scope: !10)
!23 = !DILocalVariable(name: "Param", arg: 1, scope: !24, file: !1, line: 2, type: !13)
!24 = distinct !DISubprogram(name: "InlineFunction", linkageName: "_Z14InlineFunctioni", scope: !1, file: !1, line: 2, type: !25, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!25 = !DISubroutineType(types: !26)
!26 = !{!13, !13}
!27 = !DILocation(line: 2, column: 36, scope: !24, inlinedAt: !28)
!28 = distinct !DILocation(line: 13, column: 8, scope: !10)
!29 = !DILocalVariable(name: "Var_1", scope: !24, file: !1, line: 3, type: !13)
!30 = !DILocation(line: 3, column: 7, scope: !24, inlinedAt: !28)
!31 = !DILocation(line: 3, column: 15, scope: !24, inlinedAt: !28)
!32 = !DILocalVariable(name: "Var_2", scope: !33, file: !1, line: 5, type: !13)
!33 = distinct !DILexicalBlock(scope: !24, file: !1, line: 4, column: 3)
!34 = !DILocation(line: 5, column: 9, scope: !33, inlinedAt: !28)
!35 = !DILocation(line: 5, column: 17, scope: !33, inlinedAt: !28)
!36 = !DILocation(line: 5, column: 25, scope: !33, inlinedAt: !28)
!37 = !DILocation(line: 5, column: 23, scope: !33, inlinedAt: !28)
!38 = !DILocation(line: 6, column: 13, scope: !33, inlinedAt: !28)
!39 = !DILocation(line: 6, column: 11, scope: !33, inlinedAt: !28)
!40 = !DILocation(line: 8, column: 10, scope: !24, inlinedAt: !28)
!41 = !DILocation(line: 13, column: 5, scope: !10)
!42 = !DILocation(line: 14, column: 10, scope: !10)
!43 = !DILocation(line: 14, column: 3, scope: !10)
