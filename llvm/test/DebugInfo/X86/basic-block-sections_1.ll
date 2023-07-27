; RUN: llc -O0 %s -mtriple=x86_64 -filetype=obj -o %t && llvm-dwarfdump -debug-info -v %t | FileCheck --check-prefix=NO-SECTIONS %s
; RUN: llc -O0 %s --basic-block-sections=all --unique-basic-block-section-names -mtriple=x86_64 -filetype=obj -o %t && llvm-dwarfdump -debug-info -v %t | FileCheck --check-prefix=BB-SECTIONS %s
; RUN: llc -O0 %s --basic-block-sections=all --unique-basic-block-section-names -mtriple=x86_64 -filetype=obj -split-dwarf-file=%t.dwo -o %t && llvm-dwarfdump -debug-info -v %t | FileCheck --check-prefix=BB-SECTIONS %s
; RUN: llc -O0 %s --basic-block-sections=all -mtriple=x86_64 -o - | FileCheck --check-prefix=BB-SECTIONS-ASM %s
; RUN: llc -O0 %s -mtriple=x86_64 -filetype=obj -o %t && llvm-dwarfdump  -debug-line %t | FileCheck --check-prefix=BB-SECTIONS-LINE-TABLE %s

; From:
; 1  int foo(int a) {
; 2    if (a > 20)
; 3      return bar();
; 4    else
; 5      return baz();
; 6  }

; NO-SECTIONS: DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000 ".text.hot.")
; NO-SECTIONS: DW_AT_high_pc [DW_FORM_data4] ({{.*}})
; BB-SECTIONS: DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000)
; BB-SECTIONS-NEXT: DW_AT_ranges [DW_FORM_sec_offset]
; BB-SECTIONS-NEXT: [{{.*}}) ".text.hot._Z3fooi._Z3fooi.__part.1"
; BB-SECTIONS-NEXT: [{{.*}}) ".text.hot._Z3fooi._Z3fooi.__part.2"
; BB-SECTIONS-NEXT: [{{.*}}) ".text.hot._Z3fooi._Z3fooi.__part.3"
; BB-SECTIONS-NEXT: [{{.*}}) ".text.hot._Z3fooi"
; BB-SECTIONS-ASM: _Z3fooi:
; BB-SECTIONS-ASM: .Ltmp{{[0-9]+}}:
; BB-SECTIONS-ASM-NEXT: .loc 1 2 9 prologue_end
; BB-SECTIONS-ASM: .Ltmp{{[0-9]+}}:
; BB-SECTIONS-ASM-NEXT: .loc 1 2 7 is_stmt
; BB-SECTIONS-ASM: _Z3fooi.__part.1:
; BB-SECTIONS-ASM: .LBB_END0_{{[0-9]+}}:
; BB-SECTIONS-ASM: .size	_Z3fooi.__part.1, .LBB_END0_{{[0-9]+}}-_Z3fooi.__part.1
; BB-SECTIONS-ASM: _Z3fooi.__part.2:
; BB-SECTIONS-ASM: .LBB_END0_{{[0-9]+}}:
; BB-SECTIONS-ASM: .size	_Z3fooi.__part.2, .LBB_END0_{{[0-9]+}}-_Z3fooi.__part.2
; BB-SECTIONS-ASM: _Z3fooi.__part.3:
; BB-SECTIONS-ASM: .LBB_END0_{{[0-9]+}}:
; BB-SECTIONS-ASM: .size	_Z3fooi.__part.3, .LBB_END0_{{[0-9]+}}-_Z3fooi.__part.3
; BB-SECTIONS-ASM: .Lfunc_end0:
; BB-SECTIONS-ASM: .Ldebug_ranges0:
; BB-SECTIONS-ASM-NEXT:	.quad	_Z3fooi.__part.1
; BB-SECTIONS-ASM-NEXT:	.quad	.LBB_END0_{{[0-9]+}}
; BB-SECTIONS-ASM-NEXT:	.quad	_Z3fooi.__part.2
; BB-SECTIONS-ASM-NEXT:	.quad	.LBB_END0_{{[0-9]+}}
; BB-SECTIONS-ASM-NEXT:	.quad	_Z3fooi.__part.3
; BB-SECTIONS-ASM-NEXT:	.quad	.LBB_END0_{{[0-9]+}}
; BB-SECTIONS-ASM-NEXT:	.quad	.Lfunc_begin0
; BB-SECTIONS-ASM-NEXT:	.quad	.Lfunc_end0
; BB-SECTIONS-ASM-NEXT:	.quad	0
; BB-SECTIONS-ASM-NEXT:	.quad	0
; BB-SECTIONS-LINE-TABLE:      0x0000000000000000 1 0 1 0 0 0 is_stmt
; BB-SECTIONS-LINE-TABLE-NEXT: 0x0000000000000004 2 9 1 0 0 0 is_stmt prologue_end
; BB-SECTIONS-LINE-TABLE-NEXT: 0x0000000000000008 2 7 1 0 0 0
; BB-SECTIONS-LINE-TABLE-NEXT: 0x000000000000000a 0 7 1 0 0 0
; BB-SECTIONS-LINE-TABLE-NEXT: 0x000000000000000f 3 5 1 0 0 0 is_stmt
; BB-SECTIONS-LINE-TABLE-NEXT: 0x0000000000000015 0 5 1 0 0 0
; BB-SECTIONS-LINE-TABLE-NEXT: 0x000000000000001a 5 5 1 0 0 0 is_stmt
; BB-SECTIONS-LINE-TABLE-NEXT: 0x000000000000001e 6 1 1 0 0 0 is_stmt
; BB-SECTIONS-LINE-TABLE-NEXT: 0x0000000000000022 6 1 1 0 0 0 epilogue_begin
; BB-SECTIONS-LINE-TABLE-NEXT: 0x0000000000000024 6 1 1 0 0 0 end_sequence

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_Z3fooi(i32 %0) !dbg !7 !prof !34 !section_prefix !35 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  call void @llvm.dbg.declare(metadata ptr %3, metadata !11, metadata !DIExpression()), !dbg !12
  %4 = load i32, ptr %3, align 4, !dbg !13
  %5 = icmp sgt i32 %4, 20, !dbg !15
  br i1 %5, label %6, label %8, !dbg !16, !prof !36

6:                                                ; preds = %1
  %7 = call i32 @bar()
  store i32 %7, ptr %2, align 4, !dbg !17
  br label %10, !dbg !17

8:                                                ; preds = %1
  %9 = call i32 @baz()
  store i32 %9, ptr %2, align 4, !dbg !18
  br label %10, !dbg !18

10:                                                ; preds = %8, %6
  %11 = load i32, ptr %2, align 4, !dbg !19
  ret i32 %11, !dbg !19
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare i32 @bar()
declare i32 @baz()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !20}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 10.0.0 (git@github.com:google/llvm-propeller.git f9421ebf4b3d8b64678bf6c49d1607fdce3f50c5)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "debuginfo.cc", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!12 = !DILocation(line: 1, column: 13, scope: !7)
!13 = !DILocation(line: 2, column: 7, scope: !14)
!14 = distinct !DILexicalBlock(scope: !7, file: !1, line: 2, column: 7)
!15 = !DILocation(line: 2, column: 9, scope: !14)
!16 = !DILocation(line: 2, column: 7, scope: !7)
!17 = !DILocation(line: 3, column: 5, scope: !14)
!18 = !DILocation(line: 5, column: 5, scope: !14)
!19 = !DILocation(line: 6, column: 1, scope: !7)
!20 = !{i32 1, !"ProfileSummary", !21}
!21 = !{!22, !23, !24, !25, !26, !27, !28, !29}
!22 = !{!"ProfileFormat", !"InstrProf"}
!23 = !{!"TotalCount", i64 10000}
!24 = !{!"MaxCount", i64 10}
!25 = !{!"MaxInternalCount", i64 1}
!26 = !{!"MaxFunctionCount", i64 1000}
!27 = !{!"NumCounts", i64 3}
!28 = !{!"NumFunctions", i64 5}
!29 = !{!"DetailedSummary", !30}
!30 = !{!31, !32, !33}
!31 = !{i32 10000, i64 100, i32 1}
!32 = !{i32 999900, i64 100, i32 1}
!33 = !{i32 999999, i64 1, i32 2}
!34 = !{!"function_entry_count", i64 7000}
!35 = !{!"function_section_prefix", !"hot"}
!36 = !{!"branch_weights", i32 6999, i32 1}
