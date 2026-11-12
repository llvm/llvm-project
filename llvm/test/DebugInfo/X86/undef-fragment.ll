; RUN: llc -mtriple=x86_64-pc-linux-gnu -O2 -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s

; This reproducer is based on the following C code:
;
; typedef struct { int a; int b; } S;
;
; extern int ext(int);
;
; int foo() {
;   S s = {123, 456};
;   ext(1);
;   s.a = ext(2);
;   ext(3);
;   s.a = 789;
;   return s.b;
; }
;
; and was generated using -O2 -g -fno-inline.
;
; As a small note, the third dbg.value's value has been changed from %call1 to
; undef (it would have become undef either way, but this was done to make the
; intention of the test a bit more clear).

; Verify that a location list entry describing s.b is started at the
; non-overlapping undef debug value.

; CHECK: DW_AT_location (0x00000000
; CHECK-NEXT: {{0x[0-9a-f]+}}, [[ADDR1:0x[0-9a-f]+]]): DW_OP_constu 0x7b, DW_OP_stack_value, DW_OP_piece 0x4, DW_OP_constu 0x1c8, DW_OP_stack_value, DW_OP_piece 0x4
; CHECK-NEXT: [[ADDR1]], [[ADDR2:0x[0-9a-f]+]]): DW_OP_piece 0x4, DW_OP_constu 0x1c8, DW_OP_stack_value, DW_OP_piece 0x4
; CHECK-NEXT: [[ADDR2]], {{0x[0-9a-f]+}}): DW_OP_constu 0x315, DW_OP_stack_value, DW_OP_piece 0x4, DW_OP_constu 0x1c8, DW_OP_stack_value, DW_OP_piece 0x4)
; CHECK-NEXT: DW_AT_name    ("s")

; Function Attrs: noinline nounwind uwtable
define i32 @main() !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 123, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !18
  call void @llvm.dbg.value(metadata i32 456, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !18
  %call = tail call i32 @ext(i32 1), !dbg !19
  %call1 = tail call i32 @ext(i32 2), !dbg !20
  call void @llvm.dbg.value(metadata i32 undef, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !18
  %call2 = tail call i32 @ext(i32 3), !dbg !21
  call void @llvm.dbg.value(metadata i32 789, metadata !12, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !18
  ret i32 456, !dbg !22
}

declare i32 @ext(i32)

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "undef.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!"clang version 9.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "s", scope: !7, file: !1, line: 6, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "S", file: !1, line: 1, baseType: !14)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 1, size: 64, elements: !15)
!15 = !{!16, !17}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !1, line: 1, baseType: !10, size: 32)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !14, file: !1, line: 1, baseType: !10, size: 32, offset: 32)
!18 = !DILocation(line: 6, scope: !7)
!19 = !DILocation(line: 7, scope: !7)
!20 = !DILocation(line: 8, scope: !7)
!21 = !DILocation(line: 9, scope: !7)
!22 = !DILocation(line: 11, scope: !7)
