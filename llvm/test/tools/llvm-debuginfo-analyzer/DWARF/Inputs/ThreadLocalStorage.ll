source_filename = "ThreadLocalStorage.cpp"
target triple = "x86_64-pc-linux-gnu"

@TGlobal = dso_local thread_local global i32 0, align 4, !dbg !0
@NGlobal = dso_local global i32 1, align 4, !dbg !5
@_ZZ4testvE6TLocal = internal thread_local global i32 0, align 4, !dbg !8

define dso_local void @_Z4testv() !dbg !10 {
entry:
  %NLocal = alloca i32, align 4
  %0 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @TGlobal), !dbg !22
  store i32 1, ptr %0, align 4
    #dbg_declare(ptr %NLocal, !24, !DIExpression(), !25)
  store i32 0, ptr %NLocal, align 4, !dbg !25
  store i32 2, ptr @NGlobal, align 4
  ret void
}

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "TGlobal", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "ThreadLocalStorage.cpp", directory: "")
!4 = !{!0, !5, !8}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "NGlobal", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "TLocal", scope: !10, file: !3, line: 4, type: !7, isLocal: true, isDefinition: true)
!10 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !{}
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !DILocation(line: 5, scope: !10)
!24 = !DILocalVariable(name: "NLocal", scope: !10, file: !3, line: 7, type: !7)
!25 = !DILocation(line: 7, scope: !10)
