; RUN: llc -O0 %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump -debug-dump=loc %t.o | FileCheck %s
;
; rdar://problem/15928306
;
; Test that we can emit debug info for aggregate values that are split
; up across multiple registers by SROA.
;
;    // Compile with -O1.
;    typedef struct { long int a; int b;} S;
;
;    int foo(S s) {
;            return s.b;
;    }
;
;
; CHECK: .debug_loc contents:
;

; 0x0000000000000000 - 0x0000000000000006: rdi, piece 0x00000008, rsi, piece 0x00000004
; CHECK:            Beginning address offset: 0x0000000000000000
; CHECK:               Ending address offset: [[LTMP3:.*]]
; CHECK:                Location description: 55 93 08 54 93 04
; 0x0000000000000006 - 0x0000000000000008: rbp-8, piece 0x00000008, rax, piece 0x00000004 )
; CHECK:            Beginning address offset: [[LTMP3]]
; CHECK:               Ending address offset: [[END:.*]]
; CHECK:                Location description: 76 78 93 08 54 93 04

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i64 %s.coerce0, i32 %s.coerce1) #0 !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata i64 %s.coerce0, i64 0, metadata !20, metadata !24), !dbg !21
  call void @llvm.dbg.value(metadata i32 %s.coerce1, i64 0, metadata !22, metadata !27), !dbg !21
  ret i32 %s.coerce1, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "pieces.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 3, file: !1, scope: !5, type: !6, variables: !15)
!5 = !DIFile(filename: "pieces.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !9}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "S", line: 1, file: !1, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, line: 1, size: 128, align: 64, file: !1, elements: !11)
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 1, size: 64, align: 64, file: !1, scope: !10, baseType: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 1, size: 32, align: 32, offset: 64, file: !1, scope: !10, baseType: !8)
!15 = !{!16}
!16 = !DILocalVariable(name: "s", line: 3, arg: 1, scope: !4, file: !5, type: !9)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 1, !"Debug Info Version", i32 3}
!19 = !{!"clang version 3.5 "}
!20 = !DILocalVariable(name: "s", line: 3, arg: 1, scope: !4, file: !5, type: !9)
!21 = !DILocation(line: 3, scope: !4)
!22 = !DILocalVariable(name: "s", line: 3, arg: 1, scope: !4, file: !5, type: !9)
!23 = !DILocation(line: 4, scope: !4)
!24 = !DIExpression(DW_OP_bit_piece, 0, 64)
!25 = !{}
!27 = !DIExpression(DW_OP_bit_piece, 64, 32)
