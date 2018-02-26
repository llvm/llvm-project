; RUN: llc -filetype=obj -o - < %s | llvm-dwarfdump - --debug-loc | FileCheck %s
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
; CHECK: [0x0000000000000004, 0x0000000000000004): DW_OP_constu 0x3, DW_OP_piece 0x4, DW_OP_reg5 RDI, DW_OP_piece 0x2
; CHECK: [0x0000000000000004, 0x0000000000000014): DW_OP_constu 0x3, DW_OP_piece 0x4, DW_OP_constu 0x0, DW_OP_piece 0x4

source_filename = "test/DebugInfo/X86/PR26148.ll"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%struct.S0 = type { i16, i32 }

@a = common global %struct.S0 zeroinitializer, align 4, !dbg !0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #0
; The attributes are here to force the zero-sized range not to be at the start of
; the function, which has special interpretation in DWARF. The fact that this happens
; at all is probably an LLVM bug.

define void @fn1(i16 signext %p1) #1 !dbg !16 {
entry:
  tail call void @llvm.dbg.value(metadata i16 %p1, metadata !20, metadata !23), !dbg !24
  tail call void @llvm.dbg.declare(metadata %struct.S0* undef, metadata !21, metadata !23), !dbg !25
  tail call void @llvm.dbg.declare(metadata %struct.S0* undef, metadata !22, metadata !23), !dbg !26
  tail call void @llvm.dbg.value(metadata i32 3, metadata !22, metadata !27), !dbg !26
  tail call void @llvm.dbg.value(metadata i32 0, metadata !22, metadata !28), !dbg !26
  tail call void @llvm.dbg.value(metadata i16 %p1, metadata !21, metadata !29), !dbg !25
  tail call void @llvm.dbg.value(metadata i32 3, metadata !21, metadata !27), !dbg !25
  tail call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !28), !dbg !25
  store i32 3, i32* bitcast (%struct.S0* @a to i32*), align 4, !dbg !30
  store i32 0, i32* getelementptr inbounds (%struct.S0, %struct.S0* @a, i64 0, i32 1), align 4, !dbg !30
  ret void, !dbg !31
}

define i32 @main() !dbg !32 {
entry:
  ret i32 0, !dbg !35
}

attributes #0 = { nounwind readnone }
attributes #1 = { "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 3.9.0 (https://github.com/llvm-mirror/clang 8f258397c5afd7a708bd95770c718e81d08fb11a) (https://github.com/llvm-mirror/llvm 18481855bdfa1b4a424f81be8525db002671348d)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "small.c", directory: "/Users/kfischer/Projects/clangbug")
!4 = !{}
!5 = !{!0}
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "S0", file: !3, line: 1, size: 64, align: 32, elements: !7)
!7 = !{!8, !10}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "f0", scope: !6, file: !3, line: 2, baseType: !9, size: 16, align: 16)
!9 = !DIBasicType(name: "short", size: 16, align: 16, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "f3", scope: !6, file: !3, line: 3, baseType: !11, size: 32, align: 32, offset: 32)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"PIC Level", i32 2}
!15 = !{!"clang version 3.9.0 (https://github.com/llvm-mirror/clang 8f258397c5afd7a708bd95770c718e81d08fb11a) (https://github.com/llvm-mirror/llvm 18481855bdfa1b4a424f81be8525db002671348d)"}
!16 = distinct !DISubprogram(name: "fn1", scope: !3, file: !3, line: 5, type: !17, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !9}
!19 = !{!20, !21, !22}
!20 = !DILocalVariable(name: "p1", arg: 1, scope: !16, file: !3, line: 5, type: !9)
!21 = !DILocalVariable(name: "b", scope: !16, file: !3, line: 6, type: !6)
!22 = !DILocalVariable(name: "c", scope: !16, file: !3, line: 6, type: !6)
!23 = !DIExpression()
!24 = !DILocation(line: 5, column: 16, scope: !16)
!25 = !DILocation(line: 6, column: 13, scope: !16)
!26 = !DILocation(line: 6, column: 16, scope: !16)
!27 = !DIExpression(DW_OP_LLVM_fragment, 0, 32)
!28 = !DIExpression(DW_OP_LLVM_fragment, 32, 32)
!29 = !DIExpression(DW_OP_LLVM_fragment, 32, 16)
!30 = !DILocation(line: 8, column: 9, scope: !16)
!31 = !DILocation(line: 9, column: 1, scope: !16)
!32 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 11, type: !33, isLocal: false, isDefinition: true, scopeLine: 11, isOptimized: true, unit: !2, variables: !4)
!33 = !DISubroutineType(types: !34)
!34 = !{!11}
!35 = !DILocation(line: 11, column: 14, scope: !32)

