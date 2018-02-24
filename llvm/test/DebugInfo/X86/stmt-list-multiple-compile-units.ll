; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s
; RUN: llc -O0 %s -mtriple=x86_64-apple-darwin -filetype=obj -o %t -dwarf-version=3
; RUN: llvm-dwarfdump -v %t | FileCheck %s -check-prefix=DWARF3
; RUN: llc < %s -O0 -mtriple=x86_64-apple-macosx10.7 | FileCheck %s -check-prefix=ASM

; rdar://13067005
; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_stmt_list [DW_FORM_sec_offset]   (0x00000000)
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000000)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000010)
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000000)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000010)

; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_stmt_list [DW_FORM_sec_offset]   (0x0000003c)
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000010)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000009)
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_low_pc [DW_FORM_addr]            (0x0000000000000010)
; CHECK: DW_AT_high_pc [DW_FORM_data4]          (0x00000009)


; CHECK: .debug_line contents:
; CHECK-NEXT: debug_line[{{.*}}]
; CHECK-NEXT: Line table prologue:
; CHECK-NEXT: total_length: 0x00000038
; CHECK: file_names[  1]:
; CHECK-NEXT: name: "simple.c"
; CHECK-NEXT: dir_index: 0
; CHECK: debug_line[{{.*}}]
; CHECK-NEXT: Line table prologue:
; CHECK-NEXT: total_length: 0x00000039
; CHECK: file_names[  1]:
; CHECK-NEXT: name: "simple2.c"
; CHECK-NEXT: dir_index: 0
; CHECK-NOT: file_names

; DWARF3: .debug_info contents:
; DWARF3: DW_TAG_compile_unit
; DWARF3: DW_AT_stmt_list [DW_FORM_data4]    (0x00000000)

; DWARF3: DW_TAG_compile_unit
; DWARF3: DW_AT_stmt_list [DW_FORM_data4]   (0x0000003c)


; DWARF3: .debug_line contents:
; DWARF3-NEXT: debug_line[{{.*}}]
; DWARF3-NEXT: Line table prologue:
; DWARF3-NEXT: total_length: 0x00000038
; DWARF3: file_names[  1]:
; DWARF3-NEXT: name: "simple.c"
; DWARF3-NEXT: dir_index: 0
; DWARF3: debug_line[{{.*}}]
; DWARF3-NEXT: Line table prologue:
; DWARF3-NEXT: total_length: 0x00000039
; DWARF3: file_names[  1]:
; DWARF3-NEXT: name: "simple2.c"
; DWARF3-NEXT: dir_index: 0
; DWARF3-NOT: file_names

; PR15408
; ASM: Lcu_begin0:
; ASM-NOT: Lcu_begin
; ASM: Lset[[LT:[0-9]+]] = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
; ASM-NEXT: .long   Lset[[LT]]
; ASM: Lcu_begin1:
; ASM-NOT: Lcu_begin
; ASM: Lset[[LT:[0-9]+]] = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
; ASM-NEXT: .long   Lset[[LT]]
define i32 @test(i32 %a) nounwind uwtable ssp !dbg !5 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, i32* %a.addr, align 4, !dbg !17
  %call = call i32 @fn(i32 %0), !dbg !17
  ret i32 %call, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @fn(i32 %a) nounwind uwtable ssp !dbg !13 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !19, metadata !DIExpression()), !dbg !20
  %0 = load i32, i32* %a.addr, align 4, !dbg !21
  ret i32 %0, !dbg !21
}

!llvm.dbg.cu = !{!0, !10}
!llvm.module.flags = !{!25}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3", isOptimized: false, emissionKind: FullDebug, file: !23, enums: !1, retainedTypes: !1, globals: !1, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "test", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !23, scope: !6, type: !7, variables: !1)
!6 = !DIFile(filename: "simple.c", directory: "/private/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 (trunk 172862)", isOptimized: false, emissionKind: FullDebug, file: !24, enums: !1, retainedTypes: !1, globals: !1, imports:  !1)
!13 = distinct !DISubprogram(name: "fn", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !10, scopeLine: 1, file: !24, scope: !14, type: !7, variables: !1)
!14 = !DIFile(filename: "simple2.c", directory: "/private/tmp")
!15 = !DILocalVariable(name: "a", line: 2, arg: 1, scope: !5, file: !6, type: !9)
!16 = !DILocation(line: 2, scope: !5)
!17 = !DILocation(line: 4, scope: !18)
!18 = distinct !DILexicalBlock(line: 3, column: 0, file: !23, scope: !5)
!19 = !DILocalVariable(name: "a", line: 1, arg: 1, scope: !13, file: !14, type: !9)
!20 = !DILocation(line: 1, scope: !13)
!21 = !DILocation(line: 2, scope: !22)
!22 = distinct !DILexicalBlock(line: 1, column: 0, file: !24, scope: !13)
!23 = !DIFile(filename: "simple.c", directory: "/private/tmp")
!24 = !DIFile(filename: "simple2.c", directory: "/private/tmp")
!25 = !{i32 1, !"Debug Info Version", i32 3}
