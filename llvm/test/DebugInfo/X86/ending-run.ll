; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

; Check that the line table starts at 7, not 4, but that the first
; statement isn't until line 8.

; CHECK-NOT: 0x0000000000000000      7      0      1   0  0  is_stmt
; CHECK: 0x0000000000000000      7      0      1   0
; CHECK: 0x0000000000000004      8     18      1   0  0  is_stmt prologue_end

define i32 @callee(i32 %x) nounwind uwtable ssp !dbg !5 {
entry:
  %x.addr = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !12, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.declare(metadata i32* %y, metadata !14, metadata !DIExpression()), !dbg !16
  %0 = load i32, i32* %x.addr, align 4, !dbg !17
  %1 = load i32, i32* %x.addr, align 4, !dbg !17
  %mul = mul nsw i32 %0, %1, !dbg !17
  store i32 %mul, i32* %y, align 4, !dbg !17
  %2 = load i32, i32* %y, align 4, !dbg !18
  %sub = sub nsw i32 %2, 2, !dbg !18
  ret i32 %sub, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1 (trunk 153921) (llvm/trunk 153916)", isOptimized: false, emissionKind: FullDebug, file: !19, enums: !1, retainedTypes: !1, globals: !1, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "callee", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !0, scopeLine: 7, file: !19, scope: !6, type: !7)
!6 = !DIFile(filename: "ending-run.c", directory: "/Users/echristo/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "x", line: 5, arg: 1, scope: !5, file: !6, type: !9)
!13 = !DILocation(line: 5, column: 5, scope: !5)
!14 = !DILocalVariable(name: "y", line: 8, scope: !15, file: !6, type: !9)
!15 = distinct !DILexicalBlock(line: 7, column: 1, file: !19, scope: !5)
!16 = !DILocation(line: 8, column: 9, scope: !15)
!17 = !DILocation(line: 8, column: 18, scope: !15)
!18 = !DILocation(line: 9, column: 5, scope: !15)
!19 = !DIFile(filename: "ending-run.c", directory: "/Users/echristo/tmp")
!20 = !{i32 1, !"Debug Info Version", i32 3}
