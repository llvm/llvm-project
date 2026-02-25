; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: [[void_ty:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV: [[dbg_src:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugSource %[[#]]
; CHECK-SPIRV: [[dbg_line_sum:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugLine [[dbg_src]]
; CHECK-SPIRV: [[dbg_noline_sum:%[0-9]+]] = OpExtInst [[void_ty]] %[[#]] DebugNoLine


define dso_local i32 @sum_up_to(i32 noundef %n) #0 !dbg !10 {
entry:
  %n.addr = alloca i32, align 4
  %total = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
    #dbg_declare(ptr %n.addr, !15, !DIExpression(), !16)
    #dbg_declare(ptr %total, !17, !DIExpression(), !18)
  store i32 0, ptr %total, align 4, !dbg !18
    #dbg_declare(ptr %i, !19, !DIExpression(), !21)
  store i32 0, ptr %i, align 4, !dbg !21
  br label %for.cond, !dbg !22

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4, !dbg !23
  %1 = load i32, ptr %n.addr, align 4, !dbg !25
  %cmp = icmp slt i32 %0, %1, !dbg !26
  br i1 %cmp, label %for.body, label %for.end, !dbg !27

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %i, align 4, !dbg !28
  %3 = load i32, ptr %total, align 4, !dbg !30
  %add = add nsw i32 %3, %2, !dbg !30
  store i32 %add, ptr %total, align 4, !dbg !30
  br label %for.inc, !dbg !31

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4, !dbg !32
  %inc = add nsw i32 %4, 1, !dbg !32
  store i32 %inc, ptr %i, align 4, !dbg !32
  br label %for.cond, !dbg !33, !llvm.loop !34

for.end:                                          ; preds = %for.cond
  %5 = load i32, ptr %total, align 4, !dbg !37
  ret i32 %5, !dbg !38
}

define dso_local i32 @main() #0 !dbg !39 {
entry:
  %retval = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %result, !42, !DIExpression(), !43)
  %call = call i32 @sum_up_to(i32 noundef 10), !dbg !44
  store i32 %call, ptr %result, align 4, !dbg !43
  ret i32 0, !dbg !45
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
!10 = distinct !DISubprogram(name: "sum_up_to", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12, flags: DIFlagPublic)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, flags: DIFlagPublic)
!14 = !{}
!15 = !DILocalVariable(name: "n", arg: 1, scope: !10, file: !1, line: 1, type: !13, flags: DIFlagPublic)
!16 = !DILocation(line: 1, column: 19, scope: !10)
!17 = !DILocalVariable(name: "total", scope: !10, file: !1, line: 2, type: !13, flags: DIFlagPublic)
!18 = !DILocation(line: 2, column: 9, scope: !10)
!19 = !DILocalVariable(name: "i", scope: !20, file: !1, line: 3, type: !13, flags: DIFlagPublic)
!20 = distinct !DILexicalBlock(scope: !10, file: !1, line: 3, column: 5)
!21 = !DILocation(line: 3, column: 14, scope: !20)
!22 = !DILocation(line: 3, column: 10, scope: !20)
!23 = !DILocation(line: 3, column: 21, scope: !24)
!24 = distinct !DILexicalBlock(scope: !20, file: !1, line: 3, column: 5)
!25 = !DILocation(line: 3, column: 25, scope: !24)
!26 = !DILocation(line: 3, column: 23, scope: !24)
!27 = !DILocation(line: 3, column: 5, scope: !20)
!28 = !DILocation(line: 4, column: 18, scope: !29)
!29 = distinct !DILexicalBlock(scope: !24, file: !1, line: 3, column: 33)
!30 = !DILocation(line: 4, column: 15, scope: !29)
!31 = !DILocation(line: 5, column: 5, scope: !29)
!32 = !DILocation(line: 3, column: 28, scope: !24)
!33 = !DILocation(line: 3, column: 5, scope: !24)
!34 = distinct !{!34, !27, !35, !36}
!35 = !DILocation(line: 5, column: 5, scope: !20)
!36 = !{!"llvm.loop.mustprogress"}
!37 = !DILocation(line: 6, column: 12, scope: !10)
!38 = !DILocation(line: 6, column: 5, scope: !10)
!39 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 9, type: !40, scopeLine: 9, flags: DIFlagPublic, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!40 = !DISubroutineType(types: !41, flags: DIFlagPublic)
!41 = !{!13}
!42 = !DILocalVariable(name: "result", scope: !39, file: !1, line: 10, type: !13, flags: DIFlagPublic)
!43 = !DILocation(line: 10, column: 9, scope: !39)
!44 = !DILocation(line: 10, column: 18, scope: !39)
!45 = !DILocation(line: 11, column: 5, scope: !39)
