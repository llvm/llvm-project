; RUN: opt %s -passes=verify 2>&1 | FileCheck %s
; CHECK: invalid type
; CHECK: !20 = !DILocalVariable(name: "f", scope: !21, file: !13, line: 970, type: !14)
; CHECK: !14 = !DISubroutineType(types: !15)


%timespec.0.1.2.3.0.1.2 = type { i64, i64 }
define internal i64 @init_vdso_clock_gettime(i32, ptr nonnull) unnamed_addr !dbg !142 {
  call void @llvm.dbg.value(metadata ptr null, metadata !162, metadata !DIExpression()), !dbg !167
  ret i64 -38, !dbg !168
}
declare void @llvm.dbg.value(metadata, metadata, metadata) #0
!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "zig 0.3.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !76)
!2 = !DIFile(filename: "test", directory: ".")
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Arch", scope: !5, file: !5, line: 44, baseType: !6, size: 8, align: 8, elements: !7)
!5 = !DIFile(filename: "builtin.zig", directory: "/home/andy/.local/share/zig/stage1/builtin/ugMGxVES9OkDAffv3xhJS3KQVy0Wm1xPM3Bc6x4MBuup5aetdi5pVTrGRG2aDAn0")
!6 = !DIBasicType(name: "u7", size: 8, encoding: DW_ATE_unsigned)
!7 = !{!8}
!8 = !DIEnumerator(name: "armv8_5a", value: 0)
!76 = !{!77}
!77 = !DIGlobalVariableExpression(var: !78, expr: !DIExpression())
!78 = distinct !DIGlobalVariable(name: "arch", linkageName: "arch", scope: !5, file: !5, line: 437, type: !4, isLocal: true, isDefinition: true)
!81 = !DIFile(filename: "index.zig", directory: "/store/dev/zig/build-llvm8-debug/lib/zig/std/os/linux")
!142 = distinct !DISubprogram(name: "init_vdso_clock_gettime", scope: !81, file: !81, line: 968, type: !143, scopeLine: 968, flags: DIFlagStaticMember, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !153)
!143 = !DISubroutineType(types: !144)
!144 = !{!145}
!145 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!146 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!153 = !{!154}
!154 = !DILocalVariable(name: "clk", arg: 1, scope: !142, file: !81, line: 968, type: !146)
!162 = !DILocalVariable(name: "f", scope: !163, file: !81, line: 970, type: !143)
!163 = distinct !DILexicalBlock(scope: !164, file: !81, line: 969, column: 5)
!164 = distinct !DILexicalBlock(scope: !165, file: !81, line: 968, column: 66)
!165 = distinct !DILexicalBlock(scope: !166, file: !81, line: 968, column: 45)
!166 = distinct !DILexicalBlock(scope: !142, file: !81, line: 968, column: 35)
!167 = !DILocation(line: 970, column: 5, scope: !163)
!168 = !DILocation(line: 972, column: 28, scope: !169)
!169 = distinct !DILexicalBlock(scope: !163, file: !81, line: 970, column: 5)
