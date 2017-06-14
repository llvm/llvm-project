; RUN: opt -instcombine %s -S -o - | FileCheck %s
; Verify that the eliminated instructions (bitcast, gep, load) are salvaged into
; a DIExpression.
;
; Originally created from the following C source and then heavily isolated/reduced.
;
; struct entry {
;   struct entry *next;
; };
; void scan(struct entry *queue, struct entry *end)
; {
;   struct entry *entry;
;   for (entry = (struct entry *)((char *)(queue->next) - 8);
;        &entry->next == end;
;        entry = (struct entry *)((char *)(entry->next) - 8)) {
;   }
; }

; ModuleID = '<stdin>'
source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.entry = type { %struct.entry* }

define void @salvage_load(%struct.entry** %queue) local_unnamed_addr #0 !dbg !14 {
entry:
  %im_not_dead = alloca %struct.entry*
  %0 = load %struct.entry*, %struct.entry** %queue, align 8, !dbg !19
  %1 = load %struct.entry*, %struct.entry** %queue, align 8, !dbg !19
  call void @llvm.dbg.value(metadata %struct.entry* %1, i64 0, metadata !18, metadata !20), !dbg !19
; CHECK: define void @salvage_load
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata %struct.entry** %queue, i64 0,
; CHECK-SAME:                           metadata ![[LOAD_EXPR:[0-9]+]])
  store %struct.entry* %1, %struct.entry** %im_not_dead, align 8
  ret void, !dbg !21
}

define void @salvage_bitcast(%struct.entry* %queue) local_unnamed_addr #0 !dbg !22 {
entry:
  %im_not_dead = alloca i8*
  %0 = bitcast %struct.entry* %queue to i8*, !dbg !23
  %1 = bitcast %struct.entry* %queue to i8*, !dbg !23
  call void @llvm.dbg.value(metadata i8* %1, i64 0, metadata !24, metadata !20), !dbg !23
; CHECK: define void @salvage_bitcast
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata %struct.entry* %queue, i64 0,
; CHECK-SAME:                           metadata ![[BITCAST_EXPR:[0-9]+]])
  store i8* %1, i8** %im_not_dead, align 8
  ret void, !dbg !23
}

define void @salvage_gep0(%struct.entry* %queue, %struct.entry* %end) local_unnamed_addr #0 !dbg !25 {
entry:
  %im_not_dead = alloca %struct.entry**
  %0 = getelementptr inbounds %struct.entry, %struct.entry* %queue, i32 -1, i32 0, !dbg !26
  %1 = getelementptr inbounds %struct.entry, %struct.entry* %queue, i32 -1, i32 0, !dbg !26
  call void @llvm.dbg.value(metadata %struct.entry** %1, i64 0, metadata !27, metadata !20), !dbg !26
; CHECK: define void @salvage_gep0
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata %struct.entry* %queue, i64 0,
; CHECK-SAME:                           metadata ![[GEP0_EXPR:[0-9]+]])
  store %struct.entry** %1, %struct.entry*** %im_not_dead, align 8
  ret void, !dbg !26
}

define void @salvage_gep1(%struct.entry* %queue, %struct.entry* %end) local_unnamed_addr #0 !dbg !28 {
entry:
  %im_not_dead = alloca %struct.entry**
  %0 = getelementptr inbounds %struct.entry, %struct.entry* %queue, i32 -1, i32 0, !dbg !29
  %1 = getelementptr inbounds %struct.entry, %struct.entry* %queue, i32 -1, i32 0, !dbg !29
  call void @llvm.dbg.value(metadata %struct.entry** %1, i64 0, metadata !30, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !29
; CHECK: define void @salvage_gep1
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata %struct.entry* %queue, i64 0,
; CHECK-SAME:                           metadata ![[GEP1_EXPR:[0-9]+]])
  store %struct.entry** %1, %struct.entry*** %im_not_dead, align 8
  ret void, !dbg !29
}

define void @salvage_gep2(%struct.entry* %queue, %struct.entry* %end) local_unnamed_addr #0 !dbg !31 {
entry:
  %im_not_dead = alloca %struct.entry**
  %0 = getelementptr inbounds %struct.entry, %struct.entry* %queue, i32 -1, i32 0, !dbg !32
  %1 = getelementptr inbounds %struct.entry, %struct.entry* %queue, i32 -1, i32 0, !dbg !32
  call void @llvm.dbg.value(metadata %struct.entry** %1, i64 0, metadata !33, metadata !DIExpression(DW_OP_stack_value)), !dbg !32
; CHECK: define void @salvage_gep2
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.dbg.value(metadata %struct.entry* %queue, i64 0,
; CHECK-SAME:                           metadata ![[GEP2_EXPR:[0-9]+]])
  store %struct.entry** %1, %struct.entry*** %im_not_dead, align 8
  ret void, !dbg !32
}

; CHECK: ![[LOAD_EXPR]] = !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 0)
; CHECK: ![[BITCAST_EXPR]] = !DIExpression(DW_OP_plus_uconst, 0)
; CHECK: ![[GEP0_EXPR]] = !DIExpression(DW_OP_constu, 8, DW_OP_minus, DW_OP_plus_uconst, 0, DW_OP_stack_value)
; CHECK: ![[GEP1_EXPR]] = !DIExpression(DW_OP_constu, 8, DW_OP_minus, DW_OP_stack_value,
; CHECK-SAME:                           DW_OP_LLVM_fragment, 0, 32)
; CHECK: ![[GEP2_EXPR]] = !DIExpression(DW_OP_constu, 8, DW_OP_minus, DW_OP_stack_value)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0 (trunk 297628) (llvm/trunk 297643)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{!4, !8}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "entry", file: !1, line: 1, size: 64, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "next", scope: !5, file: !1, line: 2, baseType: !4, size: 64)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"PIC Level", i32 2}
!13 = !{!"clang version 5.0.0 (trunk 297628) (llvm/trunk 297643)"}
!14 = distinct !DISubprogram(name: "scan", scope: !1, file: !1, line: 4, type: !15, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !4, !4}
!17 = !{!18}
!18 = !DILocalVariable(name: "entry", scope: !14, file: !1, line: 6, type: !4)
!19 = !DILocation(line: 6, column: 17, scope: !14)
!20 = !DIExpression(DW_OP_plus_uconst, 0)
!21 = !DILocation(line: 11, column: 1, scope: !14)
!22 = distinct !DISubprogram(name: "scan", scope: !1, file: !1, line: 4, type: !15, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !17)
!23 = !DILocation(line: 6, column: 17, scope: !22)
!24 = !DILocalVariable(name: "entry", scope: !22, file: !1, line: 6, type: !4)
!25 = distinct !DISubprogram(name: "scan", scope: !1, file: !1, line: 4, type: !15, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !17)
!26 = !DILocation(line: 6, column: 17, scope: !25)
!27 = !DILocalVariable(name: "entry", scope: !25, file: !1, line: 6, type: !4)
!28 = distinct !DISubprogram(name: "scan", scope: !1, file: !1, line: 4, type: !15, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !17)
!29 = !DILocation(line: 6, column: 17, scope: !28)
!30 = !DILocalVariable(name: "entry", scope: !28, file: !1, line: 6, type: !4)
!31 = distinct !DISubprogram(name: "scan", scope: !1, file: !1, line: 4, type: !15, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !17)
!32 = !DILocation(line: 6, column: 17, scope: !31)
!33 = !DILocalVariable(name: "entry", scope: !31, file: !1, line: 6, type: !4)
