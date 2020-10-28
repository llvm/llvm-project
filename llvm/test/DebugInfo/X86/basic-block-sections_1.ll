; RUN: llc -O0 %s -mtriple=x86_64-* -filetype=obj -o %t && llvm-dwarfdump  -debug-info -v %t | FileCheck --check-prefix=NO-SECTIONS %s
; RUN: llc -O0 %s --basic-block-sections=all --unique-basic-block-section-names -mtriple=x86_64-* -filetype=obj -o %t && llvm-dwarfdump  -debug-info -v %t | FileCheck --check-prefix=BB-SECTIONS %s
; RUN: llc -O0 %s --basic-block-sections=all --unique-basic-block-section-names -mtriple=x86_64-* -filetype=obj -split-dwarf-file=%t.dwo -o %t && llvm-dwarfdump  -debug-info -v %t | FileCheck --check-prefix=BB-SECTIONS %s
; RUN: llc -O0 %s --basic-block-sections=all -mtriple=x86_64-* -filetype=asm -o - | FileCheck --check-prefix=BB-SECTIONS-ASM %s

; From:
; int foo(int a) {
;   if (a > 20)
;     return 2;
;   else
;     return 0;
; }

; NO-SECTIONS: DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000 ".text")
; NO-SECTIONS: DW_AT_high_pc [DW_FORM_data4] ({{.*}})
; BB-SECTIONS: DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000)
; BB-SECTIONS-NEXT: DW_AT_ranges [DW_FORM_sec_offset]
; BB-SECTIONS-NEXT: [{{.*}}) ".text._Z3fooi.1"
; BB-SECTIONS-NEXT: [{{.*}}) ".text._Z3fooi.2"
; BB-SECTIONS-NEXT: [{{.*}}) ".text._Z3fooi.3"
; BB-SECTIONS-NEXT: [{{.*}}) ".text"
; BB-SECTIONS-ASM: _Z3fooi:
; BB-SECTIONS-ASM: .Ltmp{{[0-9]+}}:
; BB-SECTIONS-ASM-NEXT: .loc 1 2 9 prologue_end
; BB-SECTIONS-ASM: .Ltmp{{[0-9]+}}:
; BB-SECTIONS-ASM-NEXT: .loc 1 2 7 is_stmt
; BB-SECTIONS-ASM: _Z3fooi.1:
; BB-SECTIONS-ASM: .LBB_END0_{{[0-9]+}}:
; BB-SECTIONS-ASM: .size	_Z3fooi.1, .LBB_END0_{{[0-9]+}}-_Z3fooi.1
; BB-SECTIONS-ASM: _Z3fooi.2:
; BB-SECTIONS-ASM: .Ltmp{{[0-9]+}}:
; BB-SECTIONS-ASM-NEXT: .LBB_END0_{{[0-9]+}}:
; BB-SECTIONS-ASM: .size	_Z3fooi.2, .LBB_END0_{{[0-9]+}}-_Z3fooi.2
; BB-SECTIONS-ASM: _Z3fooi.3:
; BB-SECTIONS-ASM: .Ltmp{{[0-9]+}}:
; BB-SECTIONS-ASM-NEXT: .LBB_END0_{{[0-9]+}}:
; BB-SECTIONS-ASM: .size	_Z3fooi.3, .LBB_END0_{{[0-9]+}}-_Z3fooi.3
; BB-SECTIONS-ASM: .Lfunc_end0:
; BB-SECTIONS-ASM: .Ldebug_ranges0:
; BB-SECTIONS-ASM-NEXT:	.quad	_Z3fooi.1
; BB-SECTIONS-ASM-NEXT:	.quad	.LBB_END0_{{[0-9]+}}
; BB-SECTIONS-ASM-NEXT:	.quad	_Z3fooi.2
; BB-SECTIONS-ASM-NEXT:	.quad	.LBB_END0_{{[0-9]+}}
; BB-SECTIONS-ASM-NEXT:	.quad	_Z3fooi.3
; BB-SECTIONS-ASM-NEXT:	.quad	.LBB_END0_{{[0-9]+}}
; BB-SECTIONS-ASM-NEXT:	.quad	.Lfunc_begin0
; BB-SECTIONS-ASM-NEXT:	.quad	.Lfunc_end0
; BB-SECTIONS-ASM-NEXT:	.quad	0
; BB-SECTIONS-ASM-NEXT:	.quad	0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @_Z3fooi(i32 %0) !dbg !7 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  call void @llvm.dbg.declare(metadata i32* %3, metadata !11, metadata !DIExpression()), !dbg !12
  %4 = load i32, i32* %3, align 4, !dbg !13
  %5 = icmp sgt i32 %4, 20, !dbg !15
  br i1 %5, label %6, label %7, !dbg !16

6:                                                ; preds = %1
  store i32 2, i32* %2, align 4, !dbg !17
  br label %8, !dbg !17

7:                                                ; preds = %1
  store i32 0, i32* %2, align 4, !dbg !18
  br label %8, !dbg !18

8:                                                ; preds = %7, %6
  %9 = load i32, i32* %2, align 4, !dbg !19
  ret i32 %9, !dbg !19
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

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
