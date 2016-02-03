; RUN: llc -filetype=obj -o - < %s | llvm-dwarfdump - | FileCheck %s
;
; Created using clang -g -O3 from:
; struct S0 {
;  short f0;
;  int f3;
; } a;
; void fn1(short p1) {
;  struct S0 b, c = {3};
;  b.f3 = p1;
;  a = b = c;
; }
; 
; int main() { return 0; }
;
; This is similar to the bug in test/DebugInfo/ARM/PR26163.ll, except that there is an
; extra non-overlapping range first. Thus, we make sure that the backend actually looks
; at all expressions when determining whether to merge ranges, not just the first one.
; AS in 26163, we expect two ranges (as opposed to one), the first one being zero sized
;
;
; CHECK: 0x00000000: Beginning address offset: 0x0000000000000004
; CHECK:                Ending address offset: 0x0000000000000004
; CHECK:                 Location description: 10 03 55 93 04
; CHECK:             Beginning address offset: 0x0000000000000004
; CHECK:                Ending address offset: 0x0000000000000014
; CHECK:                 Location description: 10 03 10 00

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%struct.S0 = type { i16, i32 }

@a = common global %struct.S0 zeroinitializer, align 4

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

; The attributes are here to force the zero-sized range not to be at the start of
; the function, which has special interpretation in DWARF. The fact that this happens
; at all is probably an LLVM bug.
attributes #0 = { "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
define void @fn1(i16 signext %p1) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i16 %p1, i64 0, metadata !9, metadata !26), !dbg !27
  tail call void @llvm.dbg.declare(metadata %struct.S0* undef, metadata !10, metadata !26), !dbg !28
  tail call void @llvm.dbg.declare(metadata %struct.S0* undef, metadata !16, metadata !26), !dbg !29
  tail call void @llvm.dbg.value(metadata i32 3, i64 0, metadata !16, metadata !30), !dbg !29
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !16, metadata !31), !dbg !29
  tail call void @llvm.dbg.value(metadata i16 %p1, i64 0, metadata !10, metadata !32), !dbg !28
  tail call void @llvm.dbg.value(metadata i32 3, i64 0, metadata !10, metadata !30), !dbg !28
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !10, metadata !31), !dbg !28
  store i32 3, i32* bitcast (%struct.S0* @a to i32*), align 4, !dbg !33
  store i32 0, i32* getelementptr inbounds (%struct.S0, %struct.S0* @a, i64 0, i32 1), align 4, !dbg !33
  ret void, !dbg !34
}

define i32 @main() !dbg !17 {
entry:
  ret i32 0, !dbg !35
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23, !24}
!llvm.ident = !{!25}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (https://github.com/llvm-mirror/clang 8f258397c5afd7a708bd95770c718e81d08fb11a) (https://github.com/llvm-mirror/llvm 18481855bdfa1b4a424f81be8525db002671348d)", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3, globals: !20)
!1 = !DIFile(filename: "small.c", directory: "/Users/kfischer/Projects/clangbug")
!2 = !{}
!3 = !{!4, !17}
!4 = distinct !DISubprogram(name: "fn1", scope: !1, file: !1, line: 5, type: !5, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "short", size: 16, align: 16, encoding: DW_ATE_signed)
!8 = !{!9, !10, !16}
!9 = !DILocalVariable(name: "p1", arg: 1, scope: !4, file: !1, line: 5, type: !7)
!10 = !DILocalVariable(name: "b", scope: !4, file: !1, line: 6, type: !11)
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "S0", file: !1, line: 1, size: 64, align: 32, elements: !12)
!12 = !{!13, !14}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "f0", scope: !11, file: !1, line: 2, baseType: !7, size: 16, align: 16)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "f3", scope: !11, file: !1, line: 3, baseType: !15, size: 32, align: 32, offset: 32)
!15 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!16 = !DILocalVariable(name: "c", scope: !4, file: !1, line: 6, type: !11)
!17 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 11, type: !18, isLocal: false, isDefinition: true, scopeLine: 11, isOptimized: true, variables: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{!15}
!20 = !{!21}
!21 = !DIGlobalVariable(name: "a", scope: !0, file: !1, line: 4, type: !11, isLocal: false, isDefinition: true, variable: %struct.S0* @a)
!22 = !{i32 2, !"Dwarf Version", i32 2}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{i32 1, !"PIC Level", i32 2}
!25 = !{!"clang version 3.9.0 (https://github.com/llvm-mirror/clang 8f258397c5afd7a708bd95770c718e81d08fb11a) (https://github.com/llvm-mirror/llvm 18481855bdfa1b4a424f81be8525db002671348d)"}
!26 = !DIExpression()
!27 = !DILocation(line: 5, column: 16, scope: !4)
!28 = !DILocation(line: 6, column: 13, scope: !4)
!29 = !DILocation(line: 6, column: 16, scope: !4)
!30 = !DIExpression(DW_OP_bit_piece, 0, 32)
!31 = !DIExpression(DW_OP_bit_piece, 32, 32)
!32 = !DIExpression(DW_OP_bit_piece, 32, 16)
!33 = !DILocation(line: 8, column: 9, scope: !4)
!34 = !DILocation(line: 9, column: 1, scope: !4)
!35 = !DILocation(line: 11, column: 14, scope: !17)
