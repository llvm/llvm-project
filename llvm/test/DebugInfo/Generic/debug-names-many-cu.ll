; XFAIL: target={{.*}}-aix{{.*}}
; RUN: %llc_dwarf -accel-tables=Dwarf -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s


; Check the header
; CHECK: CU count: 257
; CHECK: Local TU count: 0
; CHECK: Foreign TU count: 0
; CHECK: Name count: 257
; CHECK: CU[0]: 0x{{[0-9a-f]*}}
; CHECK: CU[1]: 0x{{[0-9a-f]*}}
; ...
; CHECK: CU[256]: 0x{{[0-9a-f]*}}

; CHECK: Abbreviation [[ABBREV:0x[0-9a-f]*]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_compile_unit: DW_FORM_data2
; CHECK-NEXT: DW_IDX_die_offset: DW_FORM_ref4

; CHECK: String: 0x{{[0-9a-f]*}} "foobar1"
; CHECK-NEXT: Entry
; CHECK-NEXT: Abbrev: [[ABBREV]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_compile_unit: 0x0000
; CHECK-NEXT: DW_IDX_die_offset: 0x{{[0-9a-f]*}}

; CHECK: String: 0x{{[0-9a-f]*}} "foobar257"
; CHECK-NEXT: Entry
; CHECK-NEXT: Abbrev: [[ABBREV]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_compile_unit: 0x0100
; CHECK-NEXT: DW_IDX_die_offset: 0x{{[0-9a-f]*}}

; VERIFY: No errors.

!llvm.dbg.cu = !{!12, !22, !32, !42, !52, !62, !72, !82, !92, !102, !112, !122,
  !132, !142, !152, !162, !172, !182, !192, !202, !212, !222, !232, !242, !252,
  !262, !272, !282, !292, !302, !312, !322, !332, !342, !352, !362, !372, !382,
  !392, !402, !412, !422, !432, !442, !452, !462, !472, !482, !492, !502, !512,
  !522, !532, !542, !552, !562, !572, !582, !592, !602, !612, !622, !632, !642,
  !652, !662, !672, !682, !692, !702, !712, !722, !732, !742, !752, !762, !772,
  !782, !792, !802, !812, !822, !832, !842, !852, !862, !872, !882, !892, !902,
  !912, !922, !932, !942, !952, !962, !972, !982, !992, !1002, !1012, !1022,
  !1032, !1042, !1052, !1062, !1072, !1082, !1092, !1102, !1112, !1122, !1132,
  !1142, !1152, !1162, !1172, !1182, !1192, !1202, !1212, !1222, !1232, !1242,
  !1252, !1262, !1272, !1282, !1292, !1302, !1312, !1322, !1332, !1342, !1352,
  !1362, !1372, !1382, !1392, !1402, !1412, !1422, !1432, !1442, !1452, !1462,
  !1472, !1482, !1492, !1502, !1512, !1522, !1532, !1542, !1552, !1562, !1572,
  !1582, !1592, !1602, !1612, !1622, !1632, !1642, !1652, !1662, !1672, !1682,
  !1692, !1702, !1712, !1722, !1732, !1742, !1752, !1762, !1772, !1782, !1792,
  !1802, !1812, !1822, !1832, !1842, !1852, !1862, !1872, !1882, !1892, !1902,
  !1912, !1922, !1932, !1942, !1952, !1962, !1972, !1982, !1992, !2002, !2012,
  !2022, !2032, !2042, !2052, !2062, !2072, !2082, !2092, !2102, !2112, !2122,
  !2132, !2142, !2152, !2162, !2172, !2182, !2192, !2202, !2212, !2222, !2232,
  !2242, !2252, !2262, !2272, !2282, !2292, !2302, !2312, !2322, !2332, !2342,
  !2352, !2362, !2372, !2382, !2392, !2402, !2412, !2422, !2432, !2442, !2452,
  !2462, !2472, !2482, !2492, !2502, !2512, !2522, !2532, !2542, !2552, !2562,
  !2572 }
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!0}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!0 = !{!"clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)"}
!4 = !{}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!3 = !DIFile(filename: "/tmp/cu2.c", directory: "/tmp")

@foobar1 = common dso_local global ptr null, align 8, !dbg !10
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "foobar1", scope: !12, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!12 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !15)
!15 = !{!10}

@foobar2 = common dso_local global ptr null, align 8, !dbg !20
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "foobar2", scope: !22, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!22 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !25)
!25 = !{!20}

@foobar3 = common dso_local global ptr null, align 8, !dbg !30
!30 = !DIGlobalVariableExpression(var: !31, expr: !DIExpression())
!31 = distinct !DIGlobalVariable(name: "foobar3", scope: !32, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!32 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !35)
!35 = !{!30}

@foobar4 = common dso_local global ptr null, align 8, !dbg !40
!40 = !DIGlobalVariableExpression(var: !41, expr: !DIExpression())
!41 = distinct !DIGlobalVariable(name: "foobar4", scope: !42, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!42 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !45)
!45 = !{!40}

@foobar5 = common dso_local global ptr null, align 8, !dbg !50
!50 = !DIGlobalVariableExpression(var: !51, expr: !DIExpression())
!51 = distinct !DIGlobalVariable(name: "foobar5", scope: !52, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!52 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !55)
!55 = !{!50}

@foobar6 = common dso_local global ptr null, align 8, !dbg !60
!60 = !DIGlobalVariableExpression(var: !61, expr: !DIExpression())
!61 = distinct !DIGlobalVariable(name: "foobar6", scope: !62, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!62 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !65)
!65 = !{!60}

@foobar7 = common dso_local global ptr null, align 8, !dbg !70
!70 = !DIGlobalVariableExpression(var: !71, expr: !DIExpression())
!71 = distinct !DIGlobalVariable(name: "foobar7", scope: !72, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!72 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !75)
!75 = !{!70}

@foobar8 = common dso_local global ptr null, align 8, !dbg !80
!80 = !DIGlobalVariableExpression(var: !81, expr: !DIExpression())
!81 = distinct !DIGlobalVariable(name: "foobar8", scope: !82, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!82 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !85)
!85 = !{!80}

@foobar9 = common dso_local global ptr null, align 8, !dbg !90
!90 = !DIGlobalVariableExpression(var: !91, expr: !DIExpression())
!91 = distinct !DIGlobalVariable(name: "foobar9", scope: !92, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!92 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !95)
!95 = !{!90}

@foobar10 = common dso_local global ptr null, align 8, !dbg !100
!100 = !DIGlobalVariableExpression(var: !101, expr: !DIExpression())
!101 = distinct !DIGlobalVariable(name: "foobar10", scope: !102, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!102 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !105)
!105 = !{!100}

@foobar11 = common dso_local global ptr null, align 8, !dbg !110
!110 = !DIGlobalVariableExpression(var: !111, expr: !DIExpression())
!111 = distinct !DIGlobalVariable(name: "foobar11", scope: !112, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!112 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !115)
!115 = !{!110}

@foobar12 = common dso_local global ptr null, align 8, !dbg !120
!120 = !DIGlobalVariableExpression(var: !121, expr: !DIExpression())
!121 = distinct !DIGlobalVariable(name: "foobar12", scope: !122, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!122 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !125)
!125 = !{!120}

@foobar13 = common dso_local global ptr null, align 8, !dbg !130
!130 = !DIGlobalVariableExpression(var: !131, expr: !DIExpression())
!131 = distinct !DIGlobalVariable(name: "foobar13", scope: !132, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!132 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !135)
!135 = !{!130}

@foobar14 = common dso_local global ptr null, align 8, !dbg !140
!140 = !DIGlobalVariableExpression(var: !141, expr: !DIExpression())
!141 = distinct !DIGlobalVariable(name: "foobar14", scope: !142, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!142 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !145)
!145 = !{!140}

@foobar15 = common dso_local global ptr null, align 8, !dbg !150
!150 = !DIGlobalVariableExpression(var: !151, expr: !DIExpression())
!151 = distinct !DIGlobalVariable(name: "foobar15", scope: !152, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!152 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !155)
!155 = !{!150}

@foobar16 = common dso_local global ptr null, align 8, !dbg !160
!160 = !DIGlobalVariableExpression(var: !161, expr: !DIExpression())
!161 = distinct !DIGlobalVariable(name: "foobar16", scope: !162, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!162 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !165)
!165 = !{!160}

@foobar17 = common dso_local global ptr null, align 8, !dbg !170
!170 = !DIGlobalVariableExpression(var: !171, expr: !DIExpression())
!171 = distinct !DIGlobalVariable(name: "foobar17", scope: !172, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!172 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !175)
!175 = !{!170}

@foobar18 = common dso_local global ptr null, align 8, !dbg !180
!180 = !DIGlobalVariableExpression(var: !181, expr: !DIExpression())
!181 = distinct !DIGlobalVariable(name: "foobar18", scope: !182, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!182 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !185)
!185 = !{!180}

@foobar19 = common dso_local global ptr null, align 8, !dbg !190
!190 = !DIGlobalVariableExpression(var: !191, expr: !DIExpression())
!191 = distinct !DIGlobalVariable(name: "foobar19", scope: !192, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!192 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !195)
!195 = !{!190}

@foobar20 = common dso_local global ptr null, align 8, !dbg !200
!200 = !DIGlobalVariableExpression(var: !201, expr: !DIExpression())
!201 = distinct !DIGlobalVariable(name: "foobar20", scope: !202, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!202 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !205)
!205 = !{!200}

@foobar21 = common dso_local global ptr null, align 8, !dbg !210
!210 = !DIGlobalVariableExpression(var: !211, expr: !DIExpression())
!211 = distinct !DIGlobalVariable(name: "foobar21", scope: !212, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!212 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !215)
!215 = !{!210}

@foobar22 = common dso_local global ptr null, align 8, !dbg !220
!220 = !DIGlobalVariableExpression(var: !221, expr: !DIExpression())
!221 = distinct !DIGlobalVariable(name: "foobar22", scope: !222, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!222 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !225)
!225 = !{!220}

@foobar23 = common dso_local global ptr null, align 8, !dbg !230
!230 = !DIGlobalVariableExpression(var: !231, expr: !DIExpression())
!231 = distinct !DIGlobalVariable(name: "foobar23", scope: !232, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!232 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !235)
!235 = !{!230}

@foobar24 = common dso_local global ptr null, align 8, !dbg !240
!240 = !DIGlobalVariableExpression(var: !241, expr: !DIExpression())
!241 = distinct !DIGlobalVariable(name: "foobar24", scope: !242, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!242 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !245)
!245 = !{!240}

@foobar25 = common dso_local global ptr null, align 8, !dbg !250
!250 = !DIGlobalVariableExpression(var: !251, expr: !DIExpression())
!251 = distinct !DIGlobalVariable(name: "foobar25", scope: !252, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!252 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !255)
!255 = !{!250}

@foobar26 = common dso_local global ptr null, align 8, !dbg !260
!260 = !DIGlobalVariableExpression(var: !261, expr: !DIExpression())
!261 = distinct !DIGlobalVariable(name: "foobar26", scope: !262, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!262 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !265)
!265 = !{!260}

@foobar27 = common dso_local global ptr null, align 8, !dbg !270
!270 = !DIGlobalVariableExpression(var: !271, expr: !DIExpression())
!271 = distinct !DIGlobalVariable(name: "foobar27", scope: !272, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!272 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !275)
!275 = !{!270}

@foobar28 = common dso_local global ptr null, align 8, !dbg !280
!280 = !DIGlobalVariableExpression(var: !281, expr: !DIExpression())
!281 = distinct !DIGlobalVariable(name: "foobar28", scope: !282, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!282 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !285)
!285 = !{!280}

@foobar29 = common dso_local global ptr null, align 8, !dbg !290
!290 = !DIGlobalVariableExpression(var: !291, expr: !DIExpression())
!291 = distinct !DIGlobalVariable(name: "foobar29", scope: !292, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!292 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !295)
!295 = !{!290}

@foobar30 = common dso_local global ptr null, align 8, !dbg !300
!300 = !DIGlobalVariableExpression(var: !301, expr: !DIExpression())
!301 = distinct !DIGlobalVariable(name: "foobar30", scope: !302, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!302 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !305)
!305 = !{!300}

@foobar31 = common dso_local global ptr null, align 8, !dbg !310
!310 = !DIGlobalVariableExpression(var: !311, expr: !DIExpression())
!311 = distinct !DIGlobalVariable(name: "foobar31", scope: !312, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!312 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !315)
!315 = !{!310}

@foobar32 = common dso_local global ptr null, align 8, !dbg !320
!320 = !DIGlobalVariableExpression(var: !321, expr: !DIExpression())
!321 = distinct !DIGlobalVariable(name: "foobar32", scope: !322, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!322 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !325)
!325 = !{!320}

@foobar33 = common dso_local global ptr null, align 8, !dbg !330
!330 = !DIGlobalVariableExpression(var: !331, expr: !DIExpression())
!331 = distinct !DIGlobalVariable(name: "foobar33", scope: !332, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!332 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !335)
!335 = !{!330}

@foobar34 = common dso_local global ptr null, align 8, !dbg !340
!340 = !DIGlobalVariableExpression(var: !341, expr: !DIExpression())
!341 = distinct !DIGlobalVariable(name: "foobar34", scope: !342, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!342 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !345)
!345 = !{!340}

@foobar35 = common dso_local global ptr null, align 8, !dbg !350
!350 = !DIGlobalVariableExpression(var: !351, expr: !DIExpression())
!351 = distinct !DIGlobalVariable(name: "foobar35", scope: !352, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!352 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !355)
!355 = !{!350}

@foobar36 = common dso_local global ptr null, align 8, !dbg !360
!360 = !DIGlobalVariableExpression(var: !361, expr: !DIExpression())
!361 = distinct !DIGlobalVariable(name: "foobar36", scope: !362, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!362 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !365)
!365 = !{!360}

@foobar37 = common dso_local global ptr null, align 8, !dbg !370
!370 = !DIGlobalVariableExpression(var: !371, expr: !DIExpression())
!371 = distinct !DIGlobalVariable(name: "foobar37", scope: !372, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!372 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !375)
!375 = !{!370}

@foobar38 = common dso_local global ptr null, align 8, !dbg !380
!380 = !DIGlobalVariableExpression(var: !381, expr: !DIExpression())
!381 = distinct !DIGlobalVariable(name: "foobar38", scope: !382, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!382 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !385)
!385 = !{!380}

@foobar39 = common dso_local global ptr null, align 8, !dbg !390
!390 = !DIGlobalVariableExpression(var: !391, expr: !DIExpression())
!391 = distinct !DIGlobalVariable(name: "foobar39", scope: !392, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!392 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !395)
!395 = !{!390}

@foobar40 = common dso_local global ptr null, align 8, !dbg !400
!400 = !DIGlobalVariableExpression(var: !401, expr: !DIExpression())
!401 = distinct !DIGlobalVariable(name: "foobar40", scope: !402, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!402 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !405)
!405 = !{!400}

@foobar41 = common dso_local global ptr null, align 8, !dbg !410
!410 = !DIGlobalVariableExpression(var: !411, expr: !DIExpression())
!411 = distinct !DIGlobalVariable(name: "foobar41", scope: !412, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!412 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !415)
!415 = !{!410}

@foobar42 = common dso_local global ptr null, align 8, !dbg !420
!420 = !DIGlobalVariableExpression(var: !421, expr: !DIExpression())
!421 = distinct !DIGlobalVariable(name: "foobar42", scope: !422, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!422 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !425)
!425 = !{!420}

@foobar43 = common dso_local global ptr null, align 8, !dbg !430
!430 = !DIGlobalVariableExpression(var: !431, expr: !DIExpression())
!431 = distinct !DIGlobalVariable(name: "foobar43", scope: !432, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!432 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !435)
!435 = !{!430}

@foobar44 = common dso_local global ptr null, align 8, !dbg !440
!440 = !DIGlobalVariableExpression(var: !441, expr: !DIExpression())
!441 = distinct !DIGlobalVariable(name: "foobar44", scope: !442, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!442 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !445)
!445 = !{!440}

@foobar45 = common dso_local global ptr null, align 8, !dbg !450
!450 = !DIGlobalVariableExpression(var: !451, expr: !DIExpression())
!451 = distinct !DIGlobalVariable(name: "foobar45", scope: !452, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!452 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !455)
!455 = !{!450}

@foobar46 = common dso_local global ptr null, align 8, !dbg !460
!460 = !DIGlobalVariableExpression(var: !461, expr: !DIExpression())
!461 = distinct !DIGlobalVariable(name: "foobar46", scope: !462, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!462 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !465)
!465 = !{!460}

@foobar47 = common dso_local global ptr null, align 8, !dbg !470
!470 = !DIGlobalVariableExpression(var: !471, expr: !DIExpression())
!471 = distinct !DIGlobalVariable(name: "foobar47", scope: !472, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!472 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !475)
!475 = !{!470}

@foobar48 = common dso_local global ptr null, align 8, !dbg !480
!480 = !DIGlobalVariableExpression(var: !481, expr: !DIExpression())
!481 = distinct !DIGlobalVariable(name: "foobar48", scope: !482, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!482 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !485)
!485 = !{!480}

@foobar49 = common dso_local global ptr null, align 8, !dbg !490
!490 = !DIGlobalVariableExpression(var: !491, expr: !DIExpression())
!491 = distinct !DIGlobalVariable(name: "foobar49", scope: !492, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!492 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !495)
!495 = !{!490}

@foobar50 = common dso_local global ptr null, align 8, !dbg !500
!500 = !DIGlobalVariableExpression(var: !501, expr: !DIExpression())
!501 = distinct !DIGlobalVariable(name: "foobar50", scope: !502, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!502 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !505)
!505 = !{!500}

@foobar51 = common dso_local global ptr null, align 8, !dbg !510
!510 = !DIGlobalVariableExpression(var: !511, expr: !DIExpression())
!511 = distinct !DIGlobalVariable(name: "foobar51", scope: !512, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!512 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !515)
!515 = !{!510}

@foobar52 = common dso_local global ptr null, align 8, !dbg !520
!520 = !DIGlobalVariableExpression(var: !521, expr: !DIExpression())
!521 = distinct !DIGlobalVariable(name: "foobar52", scope: !522, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!522 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !525)
!525 = !{!520}

@foobar53 = common dso_local global ptr null, align 8, !dbg !530
!530 = !DIGlobalVariableExpression(var: !531, expr: !DIExpression())
!531 = distinct !DIGlobalVariable(name: "foobar53", scope: !532, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!532 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !535)
!535 = !{!530}

@foobar54 = common dso_local global ptr null, align 8, !dbg !540
!540 = !DIGlobalVariableExpression(var: !541, expr: !DIExpression())
!541 = distinct !DIGlobalVariable(name: "foobar54", scope: !542, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!542 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !545)
!545 = !{!540}

@foobar55 = common dso_local global ptr null, align 8, !dbg !550
!550 = !DIGlobalVariableExpression(var: !551, expr: !DIExpression())
!551 = distinct !DIGlobalVariable(name: "foobar55", scope: !552, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!552 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !555)
!555 = !{!550}

@foobar56 = common dso_local global ptr null, align 8, !dbg !560
!560 = !DIGlobalVariableExpression(var: !561, expr: !DIExpression())
!561 = distinct !DIGlobalVariable(name: "foobar56", scope: !562, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!562 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !565)
!565 = !{!560}

@foobar57 = common dso_local global ptr null, align 8, !dbg !570
!570 = !DIGlobalVariableExpression(var: !571, expr: !DIExpression())
!571 = distinct !DIGlobalVariable(name: "foobar57", scope: !572, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!572 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !575)
!575 = !{!570}

@foobar58 = common dso_local global ptr null, align 8, !dbg !580
!580 = !DIGlobalVariableExpression(var: !581, expr: !DIExpression())
!581 = distinct !DIGlobalVariable(name: "foobar58", scope: !582, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!582 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !585)
!585 = !{!580}

@foobar59 = common dso_local global ptr null, align 8, !dbg !590
!590 = !DIGlobalVariableExpression(var: !591, expr: !DIExpression())
!591 = distinct !DIGlobalVariable(name: "foobar59", scope: !592, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!592 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !595)
!595 = !{!590}

@foobar60 = common dso_local global ptr null, align 8, !dbg !600
!600 = !DIGlobalVariableExpression(var: !601, expr: !DIExpression())
!601 = distinct !DIGlobalVariable(name: "foobar60", scope: !602, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!602 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !605)
!605 = !{!600}

@foobar61 = common dso_local global ptr null, align 8, !dbg !610
!610 = !DIGlobalVariableExpression(var: !611, expr: !DIExpression())
!611 = distinct !DIGlobalVariable(name: "foobar61", scope: !612, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!612 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !615)
!615 = !{!610}

@foobar62 = common dso_local global ptr null, align 8, !dbg !620
!620 = !DIGlobalVariableExpression(var: !621, expr: !DIExpression())
!621 = distinct !DIGlobalVariable(name: "foobar62", scope: !622, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!622 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !625)
!625 = !{!620}

@foobar63 = common dso_local global ptr null, align 8, !dbg !630
!630 = !DIGlobalVariableExpression(var: !631, expr: !DIExpression())
!631 = distinct !DIGlobalVariable(name: "foobar63", scope: !632, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!632 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !635)
!635 = !{!630}

@foobar64 = common dso_local global ptr null, align 8, !dbg !640
!640 = !DIGlobalVariableExpression(var: !641, expr: !DIExpression())
!641 = distinct !DIGlobalVariable(name: "foobar64", scope: !642, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!642 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !645)
!645 = !{!640}

@foobar65 = common dso_local global ptr null, align 8, !dbg !650
!650 = !DIGlobalVariableExpression(var: !651, expr: !DIExpression())
!651 = distinct !DIGlobalVariable(name: "foobar65", scope: !652, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!652 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !655)
!655 = !{!650}

@foobar66 = common dso_local global ptr null, align 8, !dbg !660
!660 = !DIGlobalVariableExpression(var: !661, expr: !DIExpression())
!661 = distinct !DIGlobalVariable(name: "foobar66", scope: !662, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!662 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !665)
!665 = !{!660}

@foobar67 = common dso_local global ptr null, align 8, !dbg !670
!670 = !DIGlobalVariableExpression(var: !671, expr: !DIExpression())
!671 = distinct !DIGlobalVariable(name: "foobar67", scope: !672, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!672 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !675)
!675 = !{!670}

@foobar68 = common dso_local global ptr null, align 8, !dbg !680
!680 = !DIGlobalVariableExpression(var: !681, expr: !DIExpression())
!681 = distinct !DIGlobalVariable(name: "foobar68", scope: !682, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!682 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !685)
!685 = !{!680}

@foobar69 = common dso_local global ptr null, align 8, !dbg !690
!690 = !DIGlobalVariableExpression(var: !691, expr: !DIExpression())
!691 = distinct !DIGlobalVariable(name: "foobar69", scope: !692, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!692 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !695)
!695 = !{!690}

@foobar70 = common dso_local global ptr null, align 8, !dbg !700
!700 = !DIGlobalVariableExpression(var: !701, expr: !DIExpression())
!701 = distinct !DIGlobalVariable(name: "foobar70", scope: !702, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!702 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !705)
!705 = !{!700}

@foobar71 = common dso_local global ptr null, align 8, !dbg !710
!710 = !DIGlobalVariableExpression(var: !711, expr: !DIExpression())
!711 = distinct !DIGlobalVariable(name: "foobar71", scope: !712, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!712 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !715)
!715 = !{!710}

@foobar72 = common dso_local global ptr null, align 8, !dbg !720
!720 = !DIGlobalVariableExpression(var: !721, expr: !DIExpression())
!721 = distinct !DIGlobalVariable(name: "foobar72", scope: !722, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!722 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !725)
!725 = !{!720}

@foobar73 = common dso_local global ptr null, align 8, !dbg !730
!730 = !DIGlobalVariableExpression(var: !731, expr: !DIExpression())
!731 = distinct !DIGlobalVariable(name: "foobar73", scope: !732, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!732 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !735)
!735 = !{!730}

@foobar74 = common dso_local global ptr null, align 8, !dbg !740
!740 = !DIGlobalVariableExpression(var: !741, expr: !DIExpression())
!741 = distinct !DIGlobalVariable(name: "foobar74", scope: !742, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!742 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !745)
!745 = !{!740}

@foobar75 = common dso_local global ptr null, align 8, !dbg !750
!750 = !DIGlobalVariableExpression(var: !751, expr: !DIExpression())
!751 = distinct !DIGlobalVariable(name: "foobar75", scope: !752, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!752 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !755)
!755 = !{!750}

@foobar76 = common dso_local global ptr null, align 8, !dbg !760
!760 = !DIGlobalVariableExpression(var: !761, expr: !DIExpression())
!761 = distinct !DIGlobalVariable(name: "foobar76", scope: !762, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!762 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !765)
!765 = !{!760}

@foobar77 = common dso_local global ptr null, align 8, !dbg !770
!770 = !DIGlobalVariableExpression(var: !771, expr: !DIExpression())
!771 = distinct !DIGlobalVariable(name: "foobar77", scope: !772, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!772 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !775)
!775 = !{!770}

@foobar78 = common dso_local global ptr null, align 8, !dbg !780
!780 = !DIGlobalVariableExpression(var: !781, expr: !DIExpression())
!781 = distinct !DIGlobalVariable(name: "foobar78", scope: !782, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!782 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !785)
!785 = !{!780}

@foobar79 = common dso_local global ptr null, align 8, !dbg !790
!790 = !DIGlobalVariableExpression(var: !791, expr: !DIExpression())
!791 = distinct !DIGlobalVariable(name: "foobar79", scope: !792, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!792 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !795)
!795 = !{!790}

@foobar80 = common dso_local global ptr null, align 8, !dbg !800
!800 = !DIGlobalVariableExpression(var: !801, expr: !DIExpression())
!801 = distinct !DIGlobalVariable(name: "foobar80", scope: !802, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!802 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !805)
!805 = !{!800}

@foobar81 = common dso_local global ptr null, align 8, !dbg !810
!810 = !DIGlobalVariableExpression(var: !811, expr: !DIExpression())
!811 = distinct !DIGlobalVariable(name: "foobar81", scope: !812, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!812 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !815)
!815 = !{!810}

@foobar82 = common dso_local global ptr null, align 8, !dbg !820
!820 = !DIGlobalVariableExpression(var: !821, expr: !DIExpression())
!821 = distinct !DIGlobalVariable(name: "foobar82", scope: !822, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!822 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !825)
!825 = !{!820}

@foobar83 = common dso_local global ptr null, align 8, !dbg !830
!830 = !DIGlobalVariableExpression(var: !831, expr: !DIExpression())
!831 = distinct !DIGlobalVariable(name: "foobar83", scope: !832, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!832 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !835)
!835 = !{!830}

@foobar84 = common dso_local global ptr null, align 8, !dbg !840
!840 = !DIGlobalVariableExpression(var: !841, expr: !DIExpression())
!841 = distinct !DIGlobalVariable(name: "foobar84", scope: !842, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!842 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !845)
!845 = !{!840}

@foobar85 = common dso_local global ptr null, align 8, !dbg !850
!850 = !DIGlobalVariableExpression(var: !851, expr: !DIExpression())
!851 = distinct !DIGlobalVariable(name: "foobar85", scope: !852, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!852 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !855)
!855 = !{!850}

@foobar86 = common dso_local global ptr null, align 8, !dbg !860
!860 = !DIGlobalVariableExpression(var: !861, expr: !DIExpression())
!861 = distinct !DIGlobalVariable(name: "foobar86", scope: !862, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!862 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !865)
!865 = !{!860}

@foobar87 = common dso_local global ptr null, align 8, !dbg !870
!870 = !DIGlobalVariableExpression(var: !871, expr: !DIExpression())
!871 = distinct !DIGlobalVariable(name: "foobar87", scope: !872, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!872 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !875)
!875 = !{!870}

@foobar88 = common dso_local global ptr null, align 8, !dbg !880
!880 = !DIGlobalVariableExpression(var: !881, expr: !DIExpression())
!881 = distinct !DIGlobalVariable(name: "foobar88", scope: !882, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!882 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !885)
!885 = !{!880}

@foobar89 = common dso_local global ptr null, align 8, !dbg !890
!890 = !DIGlobalVariableExpression(var: !891, expr: !DIExpression())
!891 = distinct !DIGlobalVariable(name: "foobar89", scope: !892, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!892 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !895)
!895 = !{!890}

@foobar90 = common dso_local global ptr null, align 8, !dbg !900
!900 = !DIGlobalVariableExpression(var: !901, expr: !DIExpression())
!901 = distinct !DIGlobalVariable(name: "foobar90", scope: !902, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!902 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !905)
!905 = !{!900}

@foobar91 = common dso_local global ptr null, align 8, !dbg !910
!910 = !DIGlobalVariableExpression(var: !911, expr: !DIExpression())
!911 = distinct !DIGlobalVariable(name: "foobar91", scope: !912, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!912 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !915)
!915 = !{!910}

@foobar92 = common dso_local global ptr null, align 8, !dbg !920
!920 = !DIGlobalVariableExpression(var: !921, expr: !DIExpression())
!921 = distinct !DIGlobalVariable(name: "foobar92", scope: !922, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!922 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !925)
!925 = !{!920}

@foobar93 = common dso_local global ptr null, align 8, !dbg !930
!930 = !DIGlobalVariableExpression(var: !931, expr: !DIExpression())
!931 = distinct !DIGlobalVariable(name: "foobar93", scope: !932, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!932 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !935)
!935 = !{!930}

@foobar94 = common dso_local global ptr null, align 8, !dbg !940
!940 = !DIGlobalVariableExpression(var: !941, expr: !DIExpression())
!941 = distinct !DIGlobalVariable(name: "foobar94", scope: !942, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!942 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !945)
!945 = !{!940}

@foobar95 = common dso_local global ptr null, align 8, !dbg !950
!950 = !DIGlobalVariableExpression(var: !951, expr: !DIExpression())
!951 = distinct !DIGlobalVariable(name: "foobar95", scope: !952, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!952 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !955)
!955 = !{!950}

@foobar96 = common dso_local global ptr null, align 8, !dbg !960
!960 = !DIGlobalVariableExpression(var: !961, expr: !DIExpression())
!961 = distinct !DIGlobalVariable(name: "foobar96", scope: !962, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!962 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !965)
!965 = !{!960}

@foobar97 = common dso_local global ptr null, align 8, !dbg !970
!970 = !DIGlobalVariableExpression(var: !971, expr: !DIExpression())
!971 = distinct !DIGlobalVariable(name: "foobar97", scope: !972, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!972 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !975)
!975 = !{!970}

@foobar98 = common dso_local global ptr null, align 8, !dbg !980
!980 = !DIGlobalVariableExpression(var: !981, expr: !DIExpression())
!981 = distinct !DIGlobalVariable(name: "foobar98", scope: !982, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!982 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !985)
!985 = !{!980}

@foobar99 = common dso_local global ptr null, align 8, !dbg !990
!990 = !DIGlobalVariableExpression(var: !991, expr: !DIExpression())
!991 = distinct !DIGlobalVariable(name: "foobar99", scope: !992, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!992 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !995)
!995 = !{!990}

@foobar100 = common dso_local global ptr null, align 8, !dbg !1000
!1000 = !DIGlobalVariableExpression(var: !1001, expr: !DIExpression())
!1001 = distinct !DIGlobalVariable(name: "foobar100", scope: !1002, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1002 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1005)
!1005 = !{!1000}

@foobar101 = common dso_local global ptr null, align 8, !dbg !1010
!1010 = !DIGlobalVariableExpression(var: !1011, expr: !DIExpression())
!1011 = distinct !DIGlobalVariable(name: "foobar101", scope: !1012, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1012 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1015)
!1015 = !{!1010}

@foobar102 = common dso_local global ptr null, align 8, !dbg !1020
!1020 = !DIGlobalVariableExpression(var: !1021, expr: !DIExpression())
!1021 = distinct !DIGlobalVariable(name: "foobar102", scope: !1022, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1022 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1025)
!1025 = !{!1020}

@foobar103 = common dso_local global ptr null, align 8, !dbg !1030
!1030 = !DIGlobalVariableExpression(var: !1031, expr: !DIExpression())
!1031 = distinct !DIGlobalVariable(name: "foobar103", scope: !1032, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1032 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1035)
!1035 = !{!1030}

@foobar104 = common dso_local global ptr null, align 8, !dbg !1040
!1040 = !DIGlobalVariableExpression(var: !1041, expr: !DIExpression())
!1041 = distinct !DIGlobalVariable(name: "foobar104", scope: !1042, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1042 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1045)
!1045 = !{!1040}

@foobar105 = common dso_local global ptr null, align 8, !dbg !1050
!1050 = !DIGlobalVariableExpression(var: !1051, expr: !DIExpression())
!1051 = distinct !DIGlobalVariable(name: "foobar105", scope: !1052, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1052 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1055)
!1055 = !{!1050}

@foobar106 = common dso_local global ptr null, align 8, !dbg !1060
!1060 = !DIGlobalVariableExpression(var: !1061, expr: !DIExpression())
!1061 = distinct !DIGlobalVariable(name: "foobar106", scope: !1062, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1062 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1065)
!1065 = !{!1060}

@foobar107 = common dso_local global ptr null, align 8, !dbg !1070
!1070 = !DIGlobalVariableExpression(var: !1071, expr: !DIExpression())
!1071 = distinct !DIGlobalVariable(name: "foobar107", scope: !1072, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1072 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1075)
!1075 = !{!1070}

@foobar108 = common dso_local global ptr null, align 8, !dbg !1080
!1080 = !DIGlobalVariableExpression(var: !1081, expr: !DIExpression())
!1081 = distinct !DIGlobalVariable(name: "foobar108", scope: !1082, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1082 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1085)
!1085 = !{!1080}

@foobar109 = common dso_local global ptr null, align 8, !dbg !1090
!1090 = !DIGlobalVariableExpression(var: !1091, expr: !DIExpression())
!1091 = distinct !DIGlobalVariable(name: "foobar109", scope: !1092, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1092 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1095)
!1095 = !{!1090}

@foobar110 = common dso_local global ptr null, align 8, !dbg !1100
!1100 = !DIGlobalVariableExpression(var: !1101, expr: !DIExpression())
!1101 = distinct !DIGlobalVariable(name: "foobar110", scope: !1102, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1102 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1105)
!1105 = !{!1100}

@foobar111 = common dso_local global ptr null, align 8, !dbg !1110
!1110 = !DIGlobalVariableExpression(var: !1111, expr: !DIExpression())
!1111 = distinct !DIGlobalVariable(name: "foobar111", scope: !1112, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1112 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1115)
!1115 = !{!1110}

@foobar112 = common dso_local global ptr null, align 8, !dbg !1120
!1120 = !DIGlobalVariableExpression(var: !1121, expr: !DIExpression())
!1121 = distinct !DIGlobalVariable(name: "foobar112", scope: !1122, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1122 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1125)
!1125 = !{!1120}

@foobar113 = common dso_local global ptr null, align 8, !dbg !1130
!1130 = !DIGlobalVariableExpression(var: !1131, expr: !DIExpression())
!1131 = distinct !DIGlobalVariable(name: "foobar113", scope: !1132, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1132 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1135)
!1135 = !{!1130}

@foobar114 = common dso_local global ptr null, align 8, !dbg !1140
!1140 = !DIGlobalVariableExpression(var: !1141, expr: !DIExpression())
!1141 = distinct !DIGlobalVariable(name: "foobar114", scope: !1142, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1142 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1145)
!1145 = !{!1140}

@foobar115 = common dso_local global ptr null, align 8, !dbg !1150
!1150 = !DIGlobalVariableExpression(var: !1151, expr: !DIExpression())
!1151 = distinct !DIGlobalVariable(name: "foobar115", scope: !1152, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1152 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1155)
!1155 = !{!1150}

@foobar116 = common dso_local global ptr null, align 8, !dbg !1160
!1160 = !DIGlobalVariableExpression(var: !1161, expr: !DIExpression())
!1161 = distinct !DIGlobalVariable(name: "foobar116", scope: !1162, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1162 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1165)
!1165 = !{!1160}

@foobar117 = common dso_local global ptr null, align 8, !dbg !1170
!1170 = !DIGlobalVariableExpression(var: !1171, expr: !DIExpression())
!1171 = distinct !DIGlobalVariable(name: "foobar117", scope: !1172, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1172 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1175)
!1175 = !{!1170}

@foobar118 = common dso_local global ptr null, align 8, !dbg !1180
!1180 = !DIGlobalVariableExpression(var: !1181, expr: !DIExpression())
!1181 = distinct !DIGlobalVariable(name: "foobar118", scope: !1182, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1182 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1185)
!1185 = !{!1180}

@foobar119 = common dso_local global ptr null, align 8, !dbg !1190
!1190 = !DIGlobalVariableExpression(var: !1191, expr: !DIExpression())
!1191 = distinct !DIGlobalVariable(name: "foobar119", scope: !1192, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1192 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1195)
!1195 = !{!1190}

@foobar120 = common dso_local global ptr null, align 8, !dbg !1200
!1200 = !DIGlobalVariableExpression(var: !1201, expr: !DIExpression())
!1201 = distinct !DIGlobalVariable(name: "foobar120", scope: !1202, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1202 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1205)
!1205 = !{!1200}

@foobar121 = common dso_local global ptr null, align 8, !dbg !1210
!1210 = !DIGlobalVariableExpression(var: !1211, expr: !DIExpression())
!1211 = distinct !DIGlobalVariable(name: "foobar121", scope: !1212, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1212 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1215)
!1215 = !{!1210}

@foobar122 = common dso_local global ptr null, align 8, !dbg !1220
!1220 = !DIGlobalVariableExpression(var: !1221, expr: !DIExpression())
!1221 = distinct !DIGlobalVariable(name: "foobar122", scope: !1222, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1222 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1225)
!1225 = !{!1220}

@foobar123 = common dso_local global ptr null, align 8, !dbg !1230
!1230 = !DIGlobalVariableExpression(var: !1231, expr: !DIExpression())
!1231 = distinct !DIGlobalVariable(name: "foobar123", scope: !1232, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1232 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1235)
!1235 = !{!1230}

@foobar124 = common dso_local global ptr null, align 8, !dbg !1240
!1240 = !DIGlobalVariableExpression(var: !1241, expr: !DIExpression())
!1241 = distinct !DIGlobalVariable(name: "foobar124", scope: !1242, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1242 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1245)
!1245 = !{!1240}

@foobar125 = common dso_local global ptr null, align 8, !dbg !1250
!1250 = !DIGlobalVariableExpression(var: !1251, expr: !DIExpression())
!1251 = distinct !DIGlobalVariable(name: "foobar125", scope: !1252, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1252 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1255)
!1255 = !{!1250}

@foobar126 = common dso_local global ptr null, align 8, !dbg !1260
!1260 = !DIGlobalVariableExpression(var: !1261, expr: !DIExpression())
!1261 = distinct !DIGlobalVariable(name: "foobar126", scope: !1262, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1262 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1265)
!1265 = !{!1260}

@foobar127 = common dso_local global ptr null, align 8, !dbg !1270
!1270 = !DIGlobalVariableExpression(var: !1271, expr: !DIExpression())
!1271 = distinct !DIGlobalVariable(name: "foobar127", scope: !1272, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1272 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1275)
!1275 = !{!1270}

@foobar128 = common dso_local global ptr null, align 8, !dbg !1280
!1280 = !DIGlobalVariableExpression(var: !1281, expr: !DIExpression())
!1281 = distinct !DIGlobalVariable(name: "foobar128", scope: !1282, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1282 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1285)
!1285 = !{!1280}

@foobar129 = common dso_local global ptr null, align 8, !dbg !1290
!1290 = !DIGlobalVariableExpression(var: !1291, expr: !DIExpression())
!1291 = distinct !DIGlobalVariable(name: "foobar129", scope: !1292, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1292 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1295)
!1295 = !{!1290}

@foobar130 = common dso_local global ptr null, align 8, !dbg !1300
!1300 = !DIGlobalVariableExpression(var: !1301, expr: !DIExpression())
!1301 = distinct !DIGlobalVariable(name: "foobar130", scope: !1302, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1302 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1305)
!1305 = !{!1300}

@foobar131 = common dso_local global ptr null, align 8, !dbg !1310
!1310 = !DIGlobalVariableExpression(var: !1311, expr: !DIExpression())
!1311 = distinct !DIGlobalVariable(name: "foobar131", scope: !1312, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1312 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1315)
!1315 = !{!1310}

@foobar132 = common dso_local global ptr null, align 8, !dbg !1320
!1320 = !DIGlobalVariableExpression(var: !1321, expr: !DIExpression())
!1321 = distinct !DIGlobalVariable(name: "foobar132", scope: !1322, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1322 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1325)
!1325 = !{!1320}

@foobar133 = common dso_local global ptr null, align 8, !dbg !1330
!1330 = !DIGlobalVariableExpression(var: !1331, expr: !DIExpression())
!1331 = distinct !DIGlobalVariable(name: "foobar133", scope: !1332, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1332 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1335)
!1335 = !{!1330}

@foobar134 = common dso_local global ptr null, align 8, !dbg !1340
!1340 = !DIGlobalVariableExpression(var: !1341, expr: !DIExpression())
!1341 = distinct !DIGlobalVariable(name: "foobar134", scope: !1342, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1342 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1345)
!1345 = !{!1340}

@foobar135 = common dso_local global ptr null, align 8, !dbg !1350
!1350 = !DIGlobalVariableExpression(var: !1351, expr: !DIExpression())
!1351 = distinct !DIGlobalVariable(name: "foobar135", scope: !1352, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1352 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1355)
!1355 = !{!1350}

@foobar136 = common dso_local global ptr null, align 8, !dbg !1360
!1360 = !DIGlobalVariableExpression(var: !1361, expr: !DIExpression())
!1361 = distinct !DIGlobalVariable(name: "foobar136", scope: !1362, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1362 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1365)
!1365 = !{!1360}

@foobar137 = common dso_local global ptr null, align 8, !dbg !1370
!1370 = !DIGlobalVariableExpression(var: !1371, expr: !DIExpression())
!1371 = distinct !DIGlobalVariable(name: "foobar137", scope: !1372, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1372 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1375)
!1375 = !{!1370}

@foobar138 = common dso_local global ptr null, align 8, !dbg !1380
!1380 = !DIGlobalVariableExpression(var: !1381, expr: !DIExpression())
!1381 = distinct !DIGlobalVariable(name: "foobar138", scope: !1382, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1382 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1385)
!1385 = !{!1380}

@foobar139 = common dso_local global ptr null, align 8, !dbg !1390
!1390 = !DIGlobalVariableExpression(var: !1391, expr: !DIExpression())
!1391 = distinct !DIGlobalVariable(name: "foobar139", scope: !1392, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1392 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1395)
!1395 = !{!1390}

@foobar140 = common dso_local global ptr null, align 8, !dbg !1400
!1400 = !DIGlobalVariableExpression(var: !1401, expr: !DIExpression())
!1401 = distinct !DIGlobalVariable(name: "foobar140", scope: !1402, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1402 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1405)
!1405 = !{!1400}

@foobar141 = common dso_local global ptr null, align 8, !dbg !1410
!1410 = !DIGlobalVariableExpression(var: !1411, expr: !DIExpression())
!1411 = distinct !DIGlobalVariable(name: "foobar141", scope: !1412, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1412 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1415)
!1415 = !{!1410}

@foobar142 = common dso_local global ptr null, align 8, !dbg !1420
!1420 = !DIGlobalVariableExpression(var: !1421, expr: !DIExpression())
!1421 = distinct !DIGlobalVariable(name: "foobar142", scope: !1422, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1422 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1425)
!1425 = !{!1420}

@foobar143 = common dso_local global ptr null, align 8, !dbg !1430
!1430 = !DIGlobalVariableExpression(var: !1431, expr: !DIExpression())
!1431 = distinct !DIGlobalVariable(name: "foobar143", scope: !1432, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1432 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1435)
!1435 = !{!1430}

@foobar144 = common dso_local global ptr null, align 8, !dbg !1440
!1440 = !DIGlobalVariableExpression(var: !1441, expr: !DIExpression())
!1441 = distinct !DIGlobalVariable(name: "foobar144", scope: !1442, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1442 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1445)
!1445 = !{!1440}

@foobar145 = common dso_local global ptr null, align 8, !dbg !1450
!1450 = !DIGlobalVariableExpression(var: !1451, expr: !DIExpression())
!1451 = distinct !DIGlobalVariable(name: "foobar145", scope: !1452, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1452 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1455)
!1455 = !{!1450}

@foobar146 = common dso_local global ptr null, align 8, !dbg !1460
!1460 = !DIGlobalVariableExpression(var: !1461, expr: !DIExpression())
!1461 = distinct !DIGlobalVariable(name: "foobar146", scope: !1462, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1462 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1465)
!1465 = !{!1460}

@foobar147 = common dso_local global ptr null, align 8, !dbg !1470
!1470 = !DIGlobalVariableExpression(var: !1471, expr: !DIExpression())
!1471 = distinct !DIGlobalVariable(name: "foobar147", scope: !1472, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1472 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1475)
!1475 = !{!1470}

@foobar148 = common dso_local global ptr null, align 8, !dbg !1480
!1480 = !DIGlobalVariableExpression(var: !1481, expr: !DIExpression())
!1481 = distinct !DIGlobalVariable(name: "foobar148", scope: !1482, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1482 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1485)
!1485 = !{!1480}

@foobar149 = common dso_local global ptr null, align 8, !dbg !1490
!1490 = !DIGlobalVariableExpression(var: !1491, expr: !DIExpression())
!1491 = distinct !DIGlobalVariable(name: "foobar149", scope: !1492, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1492 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1495)
!1495 = !{!1490}

@foobar150 = common dso_local global ptr null, align 8, !dbg !1500
!1500 = !DIGlobalVariableExpression(var: !1501, expr: !DIExpression())
!1501 = distinct !DIGlobalVariable(name: "foobar150", scope: !1502, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1502 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1505)
!1505 = !{!1500}

@foobar151 = common dso_local global ptr null, align 8, !dbg !1510
!1510 = !DIGlobalVariableExpression(var: !1511, expr: !DIExpression())
!1511 = distinct !DIGlobalVariable(name: "foobar151", scope: !1512, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1512 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1515)
!1515 = !{!1510}

@foobar152 = common dso_local global ptr null, align 8, !dbg !1520
!1520 = !DIGlobalVariableExpression(var: !1521, expr: !DIExpression())
!1521 = distinct !DIGlobalVariable(name: "foobar152", scope: !1522, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1522 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1525)
!1525 = !{!1520}

@foobar153 = common dso_local global ptr null, align 8, !dbg !1530
!1530 = !DIGlobalVariableExpression(var: !1531, expr: !DIExpression())
!1531 = distinct !DIGlobalVariable(name: "foobar153", scope: !1532, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1532 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1535)
!1535 = !{!1530}

@foobar154 = common dso_local global ptr null, align 8, !dbg !1540
!1540 = !DIGlobalVariableExpression(var: !1541, expr: !DIExpression())
!1541 = distinct !DIGlobalVariable(name: "foobar154", scope: !1542, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1542 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1545)
!1545 = !{!1540}

@foobar155 = common dso_local global ptr null, align 8, !dbg !1550
!1550 = !DIGlobalVariableExpression(var: !1551, expr: !DIExpression())
!1551 = distinct !DIGlobalVariable(name: "foobar155", scope: !1552, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1552 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1555)
!1555 = !{!1550}

@foobar156 = common dso_local global ptr null, align 8, !dbg !1560
!1560 = !DIGlobalVariableExpression(var: !1561, expr: !DIExpression())
!1561 = distinct !DIGlobalVariable(name: "foobar156", scope: !1562, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1562 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1565)
!1565 = !{!1560}

@foobar157 = common dso_local global ptr null, align 8, !dbg !1570
!1570 = !DIGlobalVariableExpression(var: !1571, expr: !DIExpression())
!1571 = distinct !DIGlobalVariable(name: "foobar157", scope: !1572, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1572 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1575)
!1575 = !{!1570}

@foobar158 = common dso_local global ptr null, align 8, !dbg !1580
!1580 = !DIGlobalVariableExpression(var: !1581, expr: !DIExpression())
!1581 = distinct !DIGlobalVariable(name: "foobar158", scope: !1582, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1582 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1585)
!1585 = !{!1580}

@foobar159 = common dso_local global ptr null, align 8, !dbg !1590
!1590 = !DIGlobalVariableExpression(var: !1591, expr: !DIExpression())
!1591 = distinct !DIGlobalVariable(name: "foobar159", scope: !1592, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1592 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1595)
!1595 = !{!1590}

@foobar160 = common dso_local global ptr null, align 8, !dbg !1600
!1600 = !DIGlobalVariableExpression(var: !1601, expr: !DIExpression())
!1601 = distinct !DIGlobalVariable(name: "foobar160", scope: !1602, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1602 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1605)
!1605 = !{!1600}

@foobar161 = common dso_local global ptr null, align 8, !dbg !1610
!1610 = !DIGlobalVariableExpression(var: !1611, expr: !DIExpression())
!1611 = distinct !DIGlobalVariable(name: "foobar161", scope: !1612, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1612 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1615)
!1615 = !{!1610}

@foobar162 = common dso_local global ptr null, align 8, !dbg !1620
!1620 = !DIGlobalVariableExpression(var: !1621, expr: !DIExpression())
!1621 = distinct !DIGlobalVariable(name: "foobar162", scope: !1622, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1622 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1625)
!1625 = !{!1620}

@foobar163 = common dso_local global ptr null, align 8, !dbg !1630
!1630 = !DIGlobalVariableExpression(var: !1631, expr: !DIExpression())
!1631 = distinct !DIGlobalVariable(name: "foobar163", scope: !1632, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1632 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1635)
!1635 = !{!1630}

@foobar164 = common dso_local global ptr null, align 8, !dbg !1640
!1640 = !DIGlobalVariableExpression(var: !1641, expr: !DIExpression())
!1641 = distinct !DIGlobalVariable(name: "foobar164", scope: !1642, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1642 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1645)
!1645 = !{!1640}

@foobar165 = common dso_local global ptr null, align 8, !dbg !1650
!1650 = !DIGlobalVariableExpression(var: !1651, expr: !DIExpression())
!1651 = distinct !DIGlobalVariable(name: "foobar165", scope: !1652, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1652 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1655)
!1655 = !{!1650}

@foobar166 = common dso_local global ptr null, align 8, !dbg !1660
!1660 = !DIGlobalVariableExpression(var: !1661, expr: !DIExpression())
!1661 = distinct !DIGlobalVariable(name: "foobar166", scope: !1662, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1662 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1665)
!1665 = !{!1660}

@foobar167 = common dso_local global ptr null, align 8, !dbg !1670
!1670 = !DIGlobalVariableExpression(var: !1671, expr: !DIExpression())
!1671 = distinct !DIGlobalVariable(name: "foobar167", scope: !1672, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1672 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1675)
!1675 = !{!1670}

@foobar168 = common dso_local global ptr null, align 8, !dbg !1680
!1680 = !DIGlobalVariableExpression(var: !1681, expr: !DIExpression())
!1681 = distinct !DIGlobalVariable(name: "foobar168", scope: !1682, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1682 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1685)
!1685 = !{!1680}

@foobar169 = common dso_local global ptr null, align 8, !dbg !1690
!1690 = !DIGlobalVariableExpression(var: !1691, expr: !DIExpression())
!1691 = distinct !DIGlobalVariable(name: "foobar169", scope: !1692, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1692 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1695)
!1695 = !{!1690}

@foobar170 = common dso_local global ptr null, align 8, !dbg !1700
!1700 = !DIGlobalVariableExpression(var: !1701, expr: !DIExpression())
!1701 = distinct !DIGlobalVariable(name: "foobar170", scope: !1702, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1702 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1705)
!1705 = !{!1700}

@foobar171 = common dso_local global ptr null, align 8, !dbg !1710
!1710 = !DIGlobalVariableExpression(var: !1711, expr: !DIExpression())
!1711 = distinct !DIGlobalVariable(name: "foobar171", scope: !1712, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1712 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1715)
!1715 = !{!1710}

@foobar172 = common dso_local global ptr null, align 8, !dbg !1720
!1720 = !DIGlobalVariableExpression(var: !1721, expr: !DIExpression())
!1721 = distinct !DIGlobalVariable(name: "foobar172", scope: !1722, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1722 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1725)
!1725 = !{!1720}

@foobar173 = common dso_local global ptr null, align 8, !dbg !1730
!1730 = !DIGlobalVariableExpression(var: !1731, expr: !DIExpression())
!1731 = distinct !DIGlobalVariable(name: "foobar173", scope: !1732, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1732 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1735)
!1735 = !{!1730}

@foobar174 = common dso_local global ptr null, align 8, !dbg !1740
!1740 = !DIGlobalVariableExpression(var: !1741, expr: !DIExpression())
!1741 = distinct !DIGlobalVariable(name: "foobar174", scope: !1742, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1742 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1745)
!1745 = !{!1740}

@foobar175 = common dso_local global ptr null, align 8, !dbg !1750
!1750 = !DIGlobalVariableExpression(var: !1751, expr: !DIExpression())
!1751 = distinct !DIGlobalVariable(name: "foobar175", scope: !1752, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1752 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1755)
!1755 = !{!1750}

@foobar176 = common dso_local global ptr null, align 8, !dbg !1760
!1760 = !DIGlobalVariableExpression(var: !1761, expr: !DIExpression())
!1761 = distinct !DIGlobalVariable(name: "foobar176", scope: !1762, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1762 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1765)
!1765 = !{!1760}

@foobar177 = common dso_local global ptr null, align 8, !dbg !1770
!1770 = !DIGlobalVariableExpression(var: !1771, expr: !DIExpression())
!1771 = distinct !DIGlobalVariable(name: "foobar177", scope: !1772, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1772 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1775)
!1775 = !{!1770}

@foobar178 = common dso_local global ptr null, align 8, !dbg !1780
!1780 = !DIGlobalVariableExpression(var: !1781, expr: !DIExpression())
!1781 = distinct !DIGlobalVariable(name: "foobar178", scope: !1782, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1782 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1785)
!1785 = !{!1780}

@foobar179 = common dso_local global ptr null, align 8, !dbg !1790
!1790 = !DIGlobalVariableExpression(var: !1791, expr: !DIExpression())
!1791 = distinct !DIGlobalVariable(name: "foobar179", scope: !1792, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1792 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1795)
!1795 = !{!1790}

@foobar180 = common dso_local global ptr null, align 8, !dbg !1800
!1800 = !DIGlobalVariableExpression(var: !1801, expr: !DIExpression())
!1801 = distinct !DIGlobalVariable(name: "foobar180", scope: !1802, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1802 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1805)
!1805 = !{!1800}

@foobar181 = common dso_local global ptr null, align 8, !dbg !1810
!1810 = !DIGlobalVariableExpression(var: !1811, expr: !DIExpression())
!1811 = distinct !DIGlobalVariable(name: "foobar181", scope: !1812, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1812 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1815)
!1815 = !{!1810}

@foobar182 = common dso_local global ptr null, align 8, !dbg !1820
!1820 = !DIGlobalVariableExpression(var: !1821, expr: !DIExpression())
!1821 = distinct !DIGlobalVariable(name: "foobar182", scope: !1822, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1822 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1825)
!1825 = !{!1820}

@foobar183 = common dso_local global ptr null, align 8, !dbg !1830
!1830 = !DIGlobalVariableExpression(var: !1831, expr: !DIExpression())
!1831 = distinct !DIGlobalVariable(name: "foobar183", scope: !1832, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1832 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1835)
!1835 = !{!1830}

@foobar184 = common dso_local global ptr null, align 8, !dbg !1840
!1840 = !DIGlobalVariableExpression(var: !1841, expr: !DIExpression())
!1841 = distinct !DIGlobalVariable(name: "foobar184", scope: !1842, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1842 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1845)
!1845 = !{!1840}

@foobar185 = common dso_local global ptr null, align 8, !dbg !1850
!1850 = !DIGlobalVariableExpression(var: !1851, expr: !DIExpression())
!1851 = distinct !DIGlobalVariable(name: "foobar185", scope: !1852, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1852 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1855)
!1855 = !{!1850}

@foobar186 = common dso_local global ptr null, align 8, !dbg !1860
!1860 = !DIGlobalVariableExpression(var: !1861, expr: !DIExpression())
!1861 = distinct !DIGlobalVariable(name: "foobar186", scope: !1862, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1862 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1865)
!1865 = !{!1860}

@foobar187 = common dso_local global ptr null, align 8, !dbg !1870
!1870 = !DIGlobalVariableExpression(var: !1871, expr: !DIExpression())
!1871 = distinct !DIGlobalVariable(name: "foobar187", scope: !1872, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1872 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1875)
!1875 = !{!1870}

@foobar188 = common dso_local global ptr null, align 8, !dbg !1880
!1880 = !DIGlobalVariableExpression(var: !1881, expr: !DIExpression())
!1881 = distinct !DIGlobalVariable(name: "foobar188", scope: !1882, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1882 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1885)
!1885 = !{!1880}

@foobar189 = common dso_local global ptr null, align 8, !dbg !1890
!1890 = !DIGlobalVariableExpression(var: !1891, expr: !DIExpression())
!1891 = distinct !DIGlobalVariable(name: "foobar189", scope: !1892, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1892 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1895)
!1895 = !{!1890}

@foobar190 = common dso_local global ptr null, align 8, !dbg !1900
!1900 = !DIGlobalVariableExpression(var: !1901, expr: !DIExpression())
!1901 = distinct !DIGlobalVariable(name: "foobar190", scope: !1902, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1902 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1905)
!1905 = !{!1900}

@foobar191 = common dso_local global ptr null, align 8, !dbg !1910
!1910 = !DIGlobalVariableExpression(var: !1911, expr: !DIExpression())
!1911 = distinct !DIGlobalVariable(name: "foobar191", scope: !1912, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1912 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1915)
!1915 = !{!1910}

@foobar192 = common dso_local global ptr null, align 8, !dbg !1920
!1920 = !DIGlobalVariableExpression(var: !1921, expr: !DIExpression())
!1921 = distinct !DIGlobalVariable(name: "foobar192", scope: !1922, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1922 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1925)
!1925 = !{!1920}

@foobar193 = common dso_local global ptr null, align 8, !dbg !1930
!1930 = !DIGlobalVariableExpression(var: !1931, expr: !DIExpression())
!1931 = distinct !DIGlobalVariable(name: "foobar193", scope: !1932, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1932 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1935)
!1935 = !{!1930}

@foobar194 = common dso_local global ptr null, align 8, !dbg !1940
!1940 = !DIGlobalVariableExpression(var: !1941, expr: !DIExpression())
!1941 = distinct !DIGlobalVariable(name: "foobar194", scope: !1942, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1942 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1945)
!1945 = !{!1940}

@foobar195 = common dso_local global ptr null, align 8, !dbg !1950
!1950 = !DIGlobalVariableExpression(var: !1951, expr: !DIExpression())
!1951 = distinct !DIGlobalVariable(name: "foobar195", scope: !1952, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1952 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1955)
!1955 = !{!1950}

@foobar196 = common dso_local global ptr null, align 8, !dbg !1960
!1960 = !DIGlobalVariableExpression(var: !1961, expr: !DIExpression())
!1961 = distinct !DIGlobalVariable(name: "foobar196", scope: !1962, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1962 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1965)
!1965 = !{!1960}

@foobar197 = common dso_local global ptr null, align 8, !dbg !1970
!1970 = !DIGlobalVariableExpression(var: !1971, expr: !DIExpression())
!1971 = distinct !DIGlobalVariable(name: "foobar197", scope: !1972, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1972 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1975)
!1975 = !{!1970}

@foobar198 = common dso_local global ptr null, align 8, !dbg !1980
!1980 = !DIGlobalVariableExpression(var: !1981, expr: !DIExpression())
!1981 = distinct !DIGlobalVariable(name: "foobar198", scope: !1982, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1982 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1985)
!1985 = !{!1980}

@foobar199 = common dso_local global ptr null, align 8, !dbg !1990
!1990 = !DIGlobalVariableExpression(var: !1991, expr: !DIExpression())
!1991 = distinct !DIGlobalVariable(name: "foobar199", scope: !1992, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!1992 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !1995)
!1995 = !{!1990}

@foobar200 = common dso_local global ptr null, align 8, !dbg !2000
!2000 = !DIGlobalVariableExpression(var: !2001, expr: !DIExpression())
!2001 = distinct !DIGlobalVariable(name: "foobar200", scope: !2002, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2002 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2005)
!2005 = !{!2000}

@foobar201 = common dso_local global ptr null, align 8, !dbg !2010
!2010 = !DIGlobalVariableExpression(var: !2011, expr: !DIExpression())
!2011 = distinct !DIGlobalVariable(name: "foobar201", scope: !2012, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2012 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2015)
!2015 = !{!2010}

@foobar202 = common dso_local global ptr null, align 8, !dbg !2020
!2020 = !DIGlobalVariableExpression(var: !2021, expr: !DIExpression())
!2021 = distinct !DIGlobalVariable(name: "foobar202", scope: !2022, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2022 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2025)
!2025 = !{!2020}

@foobar203 = common dso_local global ptr null, align 8, !dbg !2030
!2030 = !DIGlobalVariableExpression(var: !2031, expr: !DIExpression())
!2031 = distinct !DIGlobalVariable(name: "foobar203", scope: !2032, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2032 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2035)
!2035 = !{!2030}

@foobar204 = common dso_local global ptr null, align 8, !dbg !2040
!2040 = !DIGlobalVariableExpression(var: !2041, expr: !DIExpression())
!2041 = distinct !DIGlobalVariable(name: "foobar204", scope: !2042, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2042 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2045)
!2045 = !{!2040}

@foobar205 = common dso_local global ptr null, align 8, !dbg !2050
!2050 = !DIGlobalVariableExpression(var: !2051, expr: !DIExpression())
!2051 = distinct !DIGlobalVariable(name: "foobar205", scope: !2052, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2052 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2055)
!2055 = !{!2050}

@foobar206 = common dso_local global ptr null, align 8, !dbg !2060
!2060 = !DIGlobalVariableExpression(var: !2061, expr: !DIExpression())
!2061 = distinct !DIGlobalVariable(name: "foobar206", scope: !2062, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2062 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2065)
!2065 = !{!2060}

@foobar207 = common dso_local global ptr null, align 8, !dbg !2070
!2070 = !DIGlobalVariableExpression(var: !2071, expr: !DIExpression())
!2071 = distinct !DIGlobalVariable(name: "foobar207", scope: !2072, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2072 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2075)
!2075 = !{!2070}

@foobar208 = common dso_local global ptr null, align 8, !dbg !2080
!2080 = !DIGlobalVariableExpression(var: !2081, expr: !DIExpression())
!2081 = distinct !DIGlobalVariable(name: "foobar208", scope: !2082, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2082 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2085)
!2085 = !{!2080}

@foobar209 = common dso_local global ptr null, align 8, !dbg !2090
!2090 = !DIGlobalVariableExpression(var: !2091, expr: !DIExpression())
!2091 = distinct !DIGlobalVariable(name: "foobar209", scope: !2092, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2092 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2095)
!2095 = !{!2090}

@foobar210 = common dso_local global ptr null, align 8, !dbg !2100
!2100 = !DIGlobalVariableExpression(var: !2101, expr: !DIExpression())
!2101 = distinct !DIGlobalVariable(name: "foobar210", scope: !2102, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2102 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2105)
!2105 = !{!2100}

@foobar211 = common dso_local global ptr null, align 8, !dbg !2110
!2110 = !DIGlobalVariableExpression(var: !2111, expr: !DIExpression())
!2111 = distinct !DIGlobalVariable(name: "foobar211", scope: !2112, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2112 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2115)
!2115 = !{!2110}

@foobar212 = common dso_local global ptr null, align 8, !dbg !2120
!2120 = !DIGlobalVariableExpression(var: !2121, expr: !DIExpression())
!2121 = distinct !DIGlobalVariable(name: "foobar212", scope: !2122, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2122 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2125)
!2125 = !{!2120}

@foobar213 = common dso_local global ptr null, align 8, !dbg !2130
!2130 = !DIGlobalVariableExpression(var: !2131, expr: !DIExpression())
!2131 = distinct !DIGlobalVariable(name: "foobar213", scope: !2132, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2132 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2135)
!2135 = !{!2130}

@foobar214 = common dso_local global ptr null, align 8, !dbg !2140
!2140 = !DIGlobalVariableExpression(var: !2141, expr: !DIExpression())
!2141 = distinct !DIGlobalVariable(name: "foobar214", scope: !2142, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2142 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2145)
!2145 = !{!2140}

@foobar215 = common dso_local global ptr null, align 8, !dbg !2150
!2150 = !DIGlobalVariableExpression(var: !2151, expr: !DIExpression())
!2151 = distinct !DIGlobalVariable(name: "foobar215", scope: !2152, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2152 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2155)
!2155 = !{!2150}

@foobar216 = common dso_local global ptr null, align 8, !dbg !2160
!2160 = !DIGlobalVariableExpression(var: !2161, expr: !DIExpression())
!2161 = distinct !DIGlobalVariable(name: "foobar216", scope: !2162, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2162 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2165)
!2165 = !{!2160}

@foobar217 = common dso_local global ptr null, align 8, !dbg !2170
!2170 = !DIGlobalVariableExpression(var: !2171, expr: !DIExpression())
!2171 = distinct !DIGlobalVariable(name: "foobar217", scope: !2172, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2172 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2175)
!2175 = !{!2170}

@foobar218 = common dso_local global ptr null, align 8, !dbg !2180
!2180 = !DIGlobalVariableExpression(var: !2181, expr: !DIExpression())
!2181 = distinct !DIGlobalVariable(name: "foobar218", scope: !2182, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2182 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2185)
!2185 = !{!2180}

@foobar219 = common dso_local global ptr null, align 8, !dbg !2190
!2190 = !DIGlobalVariableExpression(var: !2191, expr: !DIExpression())
!2191 = distinct !DIGlobalVariable(name: "foobar219", scope: !2192, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2192 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2195)
!2195 = !{!2190}

@foobar220 = common dso_local global ptr null, align 8, !dbg !2200
!2200 = !DIGlobalVariableExpression(var: !2201, expr: !DIExpression())
!2201 = distinct !DIGlobalVariable(name: "foobar220", scope: !2202, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2202 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2205)
!2205 = !{!2200}

@foobar221 = common dso_local global ptr null, align 8, !dbg !2210
!2210 = !DIGlobalVariableExpression(var: !2211, expr: !DIExpression())
!2211 = distinct !DIGlobalVariable(name: "foobar221", scope: !2212, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2212 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2215)
!2215 = !{!2210}

@foobar222 = common dso_local global ptr null, align 8, !dbg !2220
!2220 = !DIGlobalVariableExpression(var: !2221, expr: !DIExpression())
!2221 = distinct !DIGlobalVariable(name: "foobar222", scope: !2222, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2222 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2225)
!2225 = !{!2220}

@foobar223 = common dso_local global ptr null, align 8, !dbg !2230
!2230 = !DIGlobalVariableExpression(var: !2231, expr: !DIExpression())
!2231 = distinct !DIGlobalVariable(name: "foobar223", scope: !2232, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2232 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2235)
!2235 = !{!2230}

@foobar224 = common dso_local global ptr null, align 8, !dbg !2240
!2240 = !DIGlobalVariableExpression(var: !2241, expr: !DIExpression())
!2241 = distinct !DIGlobalVariable(name: "foobar224", scope: !2242, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2242 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2245)
!2245 = !{!2240}

@foobar225 = common dso_local global ptr null, align 8, !dbg !2250
!2250 = !DIGlobalVariableExpression(var: !2251, expr: !DIExpression())
!2251 = distinct !DIGlobalVariable(name: "foobar225", scope: !2252, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2252 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2255)
!2255 = !{!2250}

@foobar226 = common dso_local global ptr null, align 8, !dbg !2260
!2260 = !DIGlobalVariableExpression(var: !2261, expr: !DIExpression())
!2261 = distinct !DIGlobalVariable(name: "foobar226", scope: !2262, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2262 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2265)
!2265 = !{!2260}

@foobar227 = common dso_local global ptr null, align 8, !dbg !2270
!2270 = !DIGlobalVariableExpression(var: !2271, expr: !DIExpression())
!2271 = distinct !DIGlobalVariable(name: "foobar227", scope: !2272, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2272 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2275)
!2275 = !{!2270}

@foobar228 = common dso_local global ptr null, align 8, !dbg !2280
!2280 = !DIGlobalVariableExpression(var: !2281, expr: !DIExpression())
!2281 = distinct !DIGlobalVariable(name: "foobar228", scope: !2282, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2282 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2285)
!2285 = !{!2280}

@foobar229 = common dso_local global ptr null, align 8, !dbg !2290
!2290 = !DIGlobalVariableExpression(var: !2291, expr: !DIExpression())
!2291 = distinct !DIGlobalVariable(name: "foobar229", scope: !2292, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2292 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2295)
!2295 = !{!2290}

@foobar230 = common dso_local global ptr null, align 8, !dbg !2300
!2300 = !DIGlobalVariableExpression(var: !2301, expr: !DIExpression())
!2301 = distinct !DIGlobalVariable(name: "foobar230", scope: !2302, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2302 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2305)
!2305 = !{!2300}

@foobar231 = common dso_local global ptr null, align 8, !dbg !2310
!2310 = !DIGlobalVariableExpression(var: !2311, expr: !DIExpression())
!2311 = distinct !DIGlobalVariable(name: "foobar231", scope: !2312, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2312 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2315)
!2315 = !{!2310}

@foobar232 = common dso_local global ptr null, align 8, !dbg !2320
!2320 = !DIGlobalVariableExpression(var: !2321, expr: !DIExpression())
!2321 = distinct !DIGlobalVariable(name: "foobar232", scope: !2322, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2322 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2325)
!2325 = !{!2320}

@foobar233 = common dso_local global ptr null, align 8, !dbg !2330
!2330 = !DIGlobalVariableExpression(var: !2331, expr: !DIExpression())
!2331 = distinct !DIGlobalVariable(name: "foobar233", scope: !2332, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2332 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2335)
!2335 = !{!2330}

@foobar234 = common dso_local global ptr null, align 8, !dbg !2340
!2340 = !DIGlobalVariableExpression(var: !2341, expr: !DIExpression())
!2341 = distinct !DIGlobalVariable(name: "foobar234", scope: !2342, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2342 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2345)
!2345 = !{!2340}

@foobar235 = common dso_local global ptr null, align 8, !dbg !2350
!2350 = !DIGlobalVariableExpression(var: !2351, expr: !DIExpression())
!2351 = distinct !DIGlobalVariable(name: "foobar235", scope: !2352, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2352 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2355)
!2355 = !{!2350}

@foobar236 = common dso_local global ptr null, align 8, !dbg !2360
!2360 = !DIGlobalVariableExpression(var: !2361, expr: !DIExpression())
!2361 = distinct !DIGlobalVariable(name: "foobar236", scope: !2362, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2362 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2365)
!2365 = !{!2360}

@foobar237 = common dso_local global ptr null, align 8, !dbg !2370
!2370 = !DIGlobalVariableExpression(var: !2371, expr: !DIExpression())
!2371 = distinct !DIGlobalVariable(name: "foobar237", scope: !2372, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2372 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2375)
!2375 = !{!2370}

@foobar238 = common dso_local global ptr null, align 8, !dbg !2380
!2380 = !DIGlobalVariableExpression(var: !2381, expr: !DIExpression())
!2381 = distinct !DIGlobalVariable(name: "foobar238", scope: !2382, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2382 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2385)
!2385 = !{!2380}

@foobar239 = common dso_local global ptr null, align 8, !dbg !2390
!2390 = !DIGlobalVariableExpression(var: !2391, expr: !DIExpression())
!2391 = distinct !DIGlobalVariable(name: "foobar239", scope: !2392, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2392 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2395)
!2395 = !{!2390}

@foobar240 = common dso_local global ptr null, align 8, !dbg !2400
!2400 = !DIGlobalVariableExpression(var: !2401, expr: !DIExpression())
!2401 = distinct !DIGlobalVariable(name: "foobar240", scope: !2402, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2402 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2405)
!2405 = !{!2400}

@foobar241 = common dso_local global ptr null, align 8, !dbg !2410
!2410 = !DIGlobalVariableExpression(var: !2411, expr: !DIExpression())
!2411 = distinct !DIGlobalVariable(name: "foobar241", scope: !2412, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2412 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2415)
!2415 = !{!2410}

@foobar242 = common dso_local global ptr null, align 8, !dbg !2420
!2420 = !DIGlobalVariableExpression(var: !2421, expr: !DIExpression())
!2421 = distinct !DIGlobalVariable(name: "foobar242", scope: !2422, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2422 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2425)
!2425 = !{!2420}

@foobar243 = common dso_local global ptr null, align 8, !dbg !2430
!2430 = !DIGlobalVariableExpression(var: !2431, expr: !DIExpression())
!2431 = distinct !DIGlobalVariable(name: "foobar243", scope: !2432, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2432 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2435)
!2435 = !{!2430}

@foobar244 = common dso_local global ptr null, align 8, !dbg !2440
!2440 = !DIGlobalVariableExpression(var: !2441, expr: !DIExpression())
!2441 = distinct !DIGlobalVariable(name: "foobar244", scope: !2442, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2442 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2445)
!2445 = !{!2440}

@foobar245 = common dso_local global ptr null, align 8, !dbg !2450
!2450 = !DIGlobalVariableExpression(var: !2451, expr: !DIExpression())
!2451 = distinct !DIGlobalVariable(name: "foobar245", scope: !2452, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2452 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2455)
!2455 = !{!2450}

@foobar246 = common dso_local global ptr null, align 8, !dbg !2460
!2460 = !DIGlobalVariableExpression(var: !2461, expr: !DIExpression())
!2461 = distinct !DIGlobalVariable(name: "foobar246", scope: !2462, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2462 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2465)
!2465 = !{!2460}

@foobar247 = common dso_local global ptr null, align 8, !dbg !2470
!2470 = !DIGlobalVariableExpression(var: !2471, expr: !DIExpression())
!2471 = distinct !DIGlobalVariable(name: "foobar247", scope: !2472, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2472 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2475)
!2475 = !{!2470}

@foobar248 = common dso_local global ptr null, align 8, !dbg !2480
!2480 = !DIGlobalVariableExpression(var: !2481, expr: !DIExpression())
!2481 = distinct !DIGlobalVariable(name: "foobar248", scope: !2482, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2482 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2485)
!2485 = !{!2480}

@foobar249 = common dso_local global ptr null, align 8, !dbg !2490
!2490 = !DIGlobalVariableExpression(var: !2491, expr: !DIExpression())
!2491 = distinct !DIGlobalVariable(name: "foobar249", scope: !2492, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2492 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2495)
!2495 = !{!2490}

@foobar250 = common dso_local global ptr null, align 8, !dbg !2500
!2500 = !DIGlobalVariableExpression(var: !2501, expr: !DIExpression())
!2501 = distinct !DIGlobalVariable(name: "foobar250", scope: !2502, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2502 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2505)
!2505 = !{!2500}

@foobar251 = common dso_local global ptr null, align 8, !dbg !2510
!2510 = !DIGlobalVariableExpression(var: !2511, expr: !DIExpression())
!2511 = distinct !DIGlobalVariable(name: "foobar251", scope: !2512, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2512 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2515)
!2515 = !{!2510}

@foobar252 = common dso_local global ptr null, align 8, !dbg !2520
!2520 = !DIGlobalVariableExpression(var: !2521, expr: !DIExpression())
!2521 = distinct !DIGlobalVariable(name: "foobar252", scope: !2522, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2522 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2525)
!2525 = !{!2520}

@foobar253 = common dso_local global ptr null, align 8, !dbg !2530
!2530 = !DIGlobalVariableExpression(var: !2531, expr: !DIExpression())
!2531 = distinct !DIGlobalVariable(name: "foobar253", scope: !2532, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2532 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2535)
!2535 = !{!2530}

@foobar254 = common dso_local global ptr null, align 8, !dbg !2540
!2540 = !DIGlobalVariableExpression(var: !2541, expr: !DIExpression())
!2541 = distinct !DIGlobalVariable(name: "foobar254", scope: !2542, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2542 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2545)
!2545 = !{!2540}

@foobar255 = common dso_local global ptr null, align 8, !dbg !2550
!2550 = !DIGlobalVariableExpression(var: !2551, expr: !DIExpression())
!2551 = distinct !DIGlobalVariable(name: "foobar255", scope: !2552, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2552 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2555)
!2555 = !{!2550}

@foobar256 = common dso_local global ptr null, align 8, !dbg !2560
!2560 = !DIGlobalVariableExpression(var: !2561, expr: !DIExpression())
!2561 = distinct !DIGlobalVariable(name: "foobar256", scope: !2562, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2562 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2565)
!2565 = !{!2560}

@foobar257 = common dso_local global ptr null, align 8, !dbg !2570
!2570 = !DIGlobalVariableExpression(var: !2571, expr: !DIExpression())
!2571 = distinct !DIGlobalVariable(name: "foobar257", scope: !2572, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2572 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !2575)
!2575 = !{!2570}
