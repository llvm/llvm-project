; RUN: llc -O0 -mtriple=x86_64-apple-darwin < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - \
; RUN:     | FileCheck %s -check-prefix=CHECK -check-prefix=DWARF4
; RUN: llc -O0 -mtriple=x86_64-apple-darwin < %s -filetype=obj -dwarf-version=3 \
; RUN:     | llvm-dwarfdump -debug-dump=info - \
; RUN:     | FileCheck %s -check-prefix=CHECK -check-prefix=DWARF3

; FIXME: The location here needs to be fixed, but llvm-dwarfdump doesn't handle
; DW_AT_location lists yet.
; DWARF4: DW_AT_location [DW_FORM_sec_offset]                      (0x00000000)

; FIXME: The location here needs to be fixed, but llvm-dwarfdump doesn't handle
; DW_AT_location lists yet.
; DWARF3: DW_AT_location [DW_FORM_data4]                      (0x00000000)

; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]  ( .debug_str[0x00000067] = "vla")

; Unfortunately llvm-dwarfdump can't unparse a list of DW_AT_locations
; right now, so we check the asm output:
; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o - -filetype=asm | FileCheck %s -check-prefix=ASM-CHECK
; vla should have a register-indirect address at one point.
; ASM-CHECK: DEBUG_VALUE: vla <- RCX
; ASM-CHECK: DW_OP_breg2

; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s --check-prefix=PRETTY-PRINT
; PRETTY-PRINT: DIExpression(DW_OP_deref, DW_OP_deref)

define void @testVLAwithSize(i32 %s) nounwind uwtable ssp !dbg !5 {
entry:
  %s.addr = alloca i32, align 4
  %saved_stack = alloca i8*
  %i = alloca i32, align 4
  store i32 %s, i32* %s.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %s.addr, metadata !10, metadata !DIExpression()), !dbg !11
  %0 = load i32, i32* %s.addr, align 4, !dbg !12
  %1 = zext i32 %0 to i64, !dbg !12
  %2 = call i8* @llvm.stacksave(), !dbg !12
  store i8* %2, i8** %saved_stack, !dbg !12
  %vla = alloca i32, i64 %1, align 16, !dbg !12
  call void @llvm.dbg.declare(metadata i32* %vla, metadata !14, metadata !30), !dbg !18
  call void @llvm.dbg.declare(metadata i32* %i, metadata !19, metadata !DIExpression()), !dbg !20
  store i32 0, i32* %i, align 4, !dbg !21
  br label %for.cond, !dbg !21

for.cond:                                         ; preds = %for.inc, %entry
  %3 = load i32, i32* %i, align 4, !dbg !21
  %4 = load i32, i32* %s.addr, align 4, !dbg !21
  %cmp = icmp slt i32 %3, %4, !dbg !21
  br i1 %cmp, label %for.body, label %for.end, !dbg !21

for.body:                                         ; preds = %for.cond
  %5 = load i32, i32* %i, align 4, !dbg !23
  %6 = load i32, i32* %i, align 4, !dbg !23
  %mul = mul nsw i32 %5, %6, !dbg !23
  %7 = load i32, i32* %i, align 4, !dbg !23
  %idxprom = sext i32 %7 to i64, !dbg !23
  %arrayidx = getelementptr inbounds i32, i32* %vla, i64 %idxprom, !dbg !23
  store i32 %mul, i32* %arrayidx, align 4, !dbg !23
  br label %for.inc, !dbg !25

for.inc:                                          ; preds = %for.body
  %8 = load i32, i32* %i, align 4, !dbg !26
  %inc = add nsw i32 %8, 1, !dbg !26
  store i32 %inc, i32* %i, align 4, !dbg !26
  br label %for.cond, !dbg !26

for.end:                                          ; preds = %for.cond
  %9 = load i8*, i8** %saved_stack, !dbg !27
  call void @llvm.stackrestore(i8* %9), !dbg !27
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare void @llvm.stackrestore(i8*) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.2 (trunk 156005) (llvm/trunk 156000)", isOptimized: false, emissionKind: 1, file: !28, enums: !1, retainedTypes: !1, subprograms: !3, globals: !1, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = distinct !DISubprogram(name: "testVLAwithSize", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 2, file: !28, scope: !6, type: !7, variables: !1)
!6 = !DIFile(filename: "bar.c", directory: "/Users/echristo/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "s", line: 1, arg: 1, scope: !5, file: !6, type: !9)
!11 = !DILocation(line: 1, column: 26, scope: !5)
!12 = !DILocation(line: 3, column: 13, scope: !13)
!13 = distinct !DILexicalBlock(line: 2, column: 1, file: !28, scope: !5)
!14 = !DILocalVariable(name: "vla", line: 3, scope: !13, file: !6, type: !15)
!15 = !DICompositeType(tag: DW_TAG_array_type, align: 32, baseType: !9, elements: !16)
!16 = !{!17}
!17 = !DISubrange(count: -1)
!18 = !DILocation(line: 3, column: 7, scope: !13)
!19 = !DILocalVariable(name: "i", line: 4, scope: !13, file: !6, type: !9)
!20 = !DILocation(line: 4, column: 7, scope: !13)
!21 = !DILocation(line: 5, column: 8, scope: !22)
!22 = distinct !DILexicalBlock(line: 5, column: 3, file: !28, scope: !13)
!23 = !DILocation(line: 6, column: 5, scope: !24)
!24 = distinct !DILexicalBlock(line: 5, column: 27, file: !28, scope: !22)
!25 = !DILocation(line: 7, column: 3, scope: !24)
!26 = !DILocation(line: 5, column: 22, scope: !22)
!27 = !DILocation(line: 8, column: 1, scope: !13)
!28 = !DIFile(filename: "bar.c", directory: "/Users/echristo/tmp")
!29 = !{i32 1, !"Debug Info Version", i32 3}
!30 = !DIExpression(DW_OP_deref, DW_OP_deref)
