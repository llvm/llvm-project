; RUN: llc < %s -mtriple=arm64-apple-darwin

; rdar://9146594

source_filename = "test/CodeGen/AArch64/arm64-2011-03-17-AsmPrinterCrash.ll"

; Function Attrs: nounwind ssp
define void @drt_vsprintf() #0 {
entry:
  %do_tab_convert = alloca i32, align 4
  br i1 undef, label %if.then24, label %if.else295, !dbg !11

if.then24:                                        ; preds = %entry
  unreachable

if.else295:                                       ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %do_tab_convert, metadata !14, metadata !16), !dbg !17
  store i32 0, i32* %do_tab_convert, align 4, !dbg !18
  unreachable
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.0 (http://llvm.org/git/clang.git git:/git/puzzlebox/clang.git/ c4d1aea01c4444eb81bdbf391f1be309127c3cf1)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2)
!1 = !DIFile(filename: "print.i", directory: "/Volumes/Ebi/echeng/radars/r9146594")
!2 = !{!3}
!3 = !DIGlobalVariableExpression(var: !4, expr: !DIExpression())
!4 = !DIGlobalVariable(name: "vsplive", scope: !5, file: !1, line: 617, type: !8, isLocal: true, isDefinition: true)
!5 = distinct !DISubprogram(name: "drt_vsprintf", scope: !1, file: !1, line: 616, type: !6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 2, !"Dwarf Version", i32 2}
!11 = !DILocation(line: 653, column: 5, scope: !12)
!12 = distinct !DILexicalBlock(scope: !13, file: !1, line: 652, column: 35)
!13 = distinct !DILexicalBlock(scope: !5, file: !1, line: 616, column: 1)
!14 = !DILocalVariable(name: "do_tab_convert", scope: !15, file: !1, line: 853, type: !8)
!15 = distinct !DILexicalBlock(scope: !12, file: !1, line: 850, column: 12)
!16 = !DIExpression()
!17 = !DILocation(line: 853, column: 11, scope: !15)
!18 = !DILocation(line: 853, column: 29, scope: !15)

