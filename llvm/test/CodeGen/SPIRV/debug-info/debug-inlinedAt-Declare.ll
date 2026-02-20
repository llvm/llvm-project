; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[STR_MAIN:[0-9]+]] = OpString "main"
; CHECK-SPIRV: %[[STR_N:[0-9]+]] = OpString "n"
; CHECK-SPIRV: %[[INT_T:[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV: %[[VOID_T:[0-9]+]] = OpTypeVoid
; CHECK-SPIRV: %[[VAL_14:[0-9]+]] = OpConstant %[[INT_T]] 14
; CHECK-SPIRV: %[[DBG_FUNC_MAIN:[0-9]+]] = OpExtInst %[[VOID_T]] %[[#]] DebugFunction %[[STR_MAIN]]
; CHECK-SPIRV: %[[DBG_LOCAL:[0-9]+]] = OpExtInst {{%[0-9]+}} %[[#]] DebugLocalVariable %[[STR_N]]
; CHECK-SPIRV: %[[DBG_INLINED_AT:[0-9]+]] = OpExtInst %[[VOID_T]] %[[#]] DebugInlinedAt %[[VAL_14]] %[[DBG_FUNC_MAIN]]
; CHECK-SPIRV: %[[DBG_DEC:[0-9]+]] = OpExtInst %[[VOID_T]] %[[#]] DebugDeclare %[[DBG_LOCAL]]


define dso_local i32 @sum_up_to(i32 noundef %0) #0 !dbg !10 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
    #dbg_declare(ptr %2, !15, !DIExpression(), !16)
    #dbg_declare(ptr %3, !17, !DIExpression(), !18)
  store i32 0, ptr %3, align 4, !dbg !18
    #dbg_declare(ptr %4, !19, !DIExpression(), !21)
  store i32 0, ptr %4, align 4, !dbg !21
  br label %5, !dbg !22

5:                                                ; preds = %13, %1
  %6 = load i32, ptr %4, align 4, !dbg !23
  %7 = load i32, ptr %2, align 4, !dbg !25
  %8 = icmp slt i32 %6, %7, !dbg !26
  br i1 %8, label %9, label %16, !dbg !27

9:                                                ; preds = %5
  %10 = load i32, ptr %4, align 4, !dbg !28
  %11 = load i32, ptr %3, align 4, !dbg !30
  %12 = add nsw i32 %11, %10, !dbg !30
  store i32 %12, ptr %3, align 4, !dbg !30
  br label %13, !dbg !31

13:                                               ; preds = %9
  %14 = load i32, ptr %4, align 4, !dbg !32
  %15 = add nsw i32 %14, 1, !dbg !32
  store i32 %15, ptr %4, align 4, !dbg !32
  br label %5, !dbg !33, !llvm.loop !34

16:                                               ; preds = %5
  %17 = load i32, ptr %3, align 4, !dbg !37
  ret i32 %17, !dbg !38
}

define dso_local i32 @main() #1 !dbg !39 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %4, align 4
    #dbg_declare(ptr %5, !42, !DIExpression(), !43)
  store i32 10, ptr %1, align 4
    #dbg_declare(ptr %1, !15, !DIExpression(), !44)
    #dbg_declare(ptr %2, !17, !DIExpression(), !46)
  store i32 0, ptr %2, align 4, !dbg !46
    #dbg_declare(ptr %3, !19, !DIExpression(), !47)
  store i32 0, ptr %3, align 4, !dbg !47
  br label %6, !dbg !48

6:                                                ; preds = %10, %0
  %7 = load i32, ptr %3, align 4, !dbg !49
  %8 = load i32, ptr %1, align 4, !dbg !50
  %9 = icmp slt i32 %7, %8, !dbg !51
  br i1 %9, label %10, label %16, !dbg !52

10:                                               ; preds = %6
  %11 = load i32, ptr %3, align 4, !dbg !53
  %12 = load i32, ptr %2, align 4, !dbg !54
  %13 = add nsw i32 %12, %11, !dbg !54
  store i32 %13, ptr %2, align 4, !dbg !54
  %14 = load i32, ptr %3, align 4, !dbg !55
  %15 = add nsw i32 %14, 1, !dbg !55
  store i32 %15, ptr %3, align 4, !dbg !55
  br label %6, !dbg !56, !llvm.loop !57

16:                                               ; preds = %6
  %17 = load i32, ptr %2, align 4, !dbg !59
  store i32 %17, ptr %5, align 4, !dbg !43
  ret i32 0, !dbg !60
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version XX.X.XXXX (FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "example.cpp", directory: "/AAAAAAAAAA/BBBBBBBB/CCCCCCCCC", checksumkind: CSK_MD5, checksum: "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!10 = distinct !DISubprogram(name: "sum_up_to", scope: !1, file: !1, line: 5, type: !11, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12, flags: DIFlagPublic)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!14 = !{}
!15 = !DILocalVariable(name: "n", arg: 1, scope: !10, file: !1, line: 5, type: !13, flags: DIFlagPublic)
!16 = !DILocation(line: 5, column: 19, scope: !10)
!17 = !DILocalVariable(name: "total", scope: !10, file: !1, line: 6, type: !13, flags: DIFlagPublic)
!18 = !DILocation(line: 6, column: 9, scope: !10)
!19 = !DILocalVariable(name: "i", scope: !20, file: !1, line: 7, type: !13, flags: DIFlagPublic)
!20 = distinct !DILexicalBlock(scope: !10, file: !1, line: 7, column: 5)
!21 = !DILocation(line: 7, column: 14, scope: !20)
!22 = !DILocation(line: 7, column: 10, scope: !20)
!23 = !DILocation(line: 7, column: 21, scope: !24)
!24 = distinct !DILexicalBlock(scope: !20, file: !1, line: 7, column: 5)
!25 = !DILocation(line: 7, column: 25, scope: !24)
!26 = !DILocation(line: 7, column: 23, scope: !24)
!27 = !DILocation(line: 7, column: 5, scope: !20)
!28 = !DILocation(line: 8, column: 18, scope: !29)
!29 = distinct !DILexicalBlock(scope: !24, file: !1, line: 7, column: 33)
!30 = !DILocation(line: 8, column: 15, scope: !29)
!31 = !DILocation(line: 9, column: 5, scope: !29)
!32 = !DILocation(line: 7, column: 28, scope: !24)
!33 = !DILocation(line: 7, column: 5, scope: !24)
!34 = distinct !{!34, !27, !35, !36}
!35 = !DILocation(line: 9, column: 5, scope: !20)
!36 = !{!"llvm.loop.mustprogress"}
!37 = !DILocation(line: 10, column: 12, scope: !10)
!38 = !DILocation(line: 10, column: 5, scope: !10)
!39 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 13, type: !40, scopeLine: 13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14, flags: DIFlagPublic)
!40 = !DISubroutineType(types: !41, flags: DIFlagPublic)
!41 = !{!13}
!42 = !DILocalVariable(name: "result", scope: !39, file: !1, line: 14, type: !13, flags: DIFlagPublic)
!43 = !DILocation(line: 14, column: 9, scope: !39)
!44 = !DILocation(line: 5, column: 19, scope: !10, inlinedAt: !45)
!45 = distinct !DILocation(line: 14, column: 18, scope: !39)
!46 = !DILocation(line: 6, column: 9, scope: !10, inlinedAt: !45)
!47 = !DILocation(line: 7, column: 14, scope: !20, inlinedAt: !45)
!48 = !DILocation(line: 7, column: 10, scope: !20, inlinedAt: !45)
!49 = !DILocation(line: 7, column: 21, scope: !24, inlinedAt: !45)
!50 = !DILocation(line: 7, column: 25, scope: !24, inlinedAt: !45)
!51 = !DILocation(line: 7, column: 23, scope: !24, inlinedAt: !45)
!52 = !DILocation(line: 7, column: 5, scope: !20, inlinedAt: !45)
!53 = !DILocation(line: 8, column: 18, scope: !29, inlinedAt: !45)
!54 = !DILocation(line: 8, column: 15, scope: !29, inlinedAt: !45)
!55 = !DILocation(line: 7, column: 28, scope: !24, inlinedAt: !45)
!56 = !DILocation(line: 7, column: 5, scope: !24, inlinedAt: !45)
!57 = distinct !{!57, !52, !58, !36}
!58 = !DILocation(line: 9, column: 5, scope: !20, inlinedAt: !45)
!59 = !DILocation(line: 10, column: 12, scope: !10, inlinedAt: !45)
!60 = !DILocation(line: 15, column: 5, scope: !39)
