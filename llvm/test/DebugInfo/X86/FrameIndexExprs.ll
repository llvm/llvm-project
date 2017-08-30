; PR31381: An assertion in the DWARF backend when fragments in MMI slots are
; sorted by largest offset first.
; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_location [DW_FORM_exprloc] (DW_OP_fbreg -8, DW_OP_piece 0x3, DW_OP_piece 0x6, DW_OP_fbreg -3, DW_OP_piece 0x3)
; CHECK-NEXT: DW_AT_abstract_origin {{.*}}"p"
source_filename = "bugpoint-reduced-simplified.ll"
target triple = "x86_64-apple-darwin"

@f = common local_unnamed_addr global i32 0, align 4, !dbg !0
@h = common local_unnamed_addr global i32 0, align 4, !dbg !6

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

define void @fn4() local_unnamed_addr !dbg !12 {
entry:
  %l1.sroa.7.i = alloca [3 x i8], align 1
  tail call void @llvm.dbg.declare(metadata [3 x i8]* %l1.sroa.7.i, metadata !15, metadata !26), !dbg !27
  %i.sroa.4.i = alloca [3 x i8], align 8
  tail call void @llvm.dbg.declare(metadata [3 x i8]* %i.sroa.4.i, metadata !15, metadata !32), !dbg !27
  %0 = load i32, i32* @h, align 4
  br label %while.body.i.i, !dbg !33

while.body.i.i:                                   ; preds = %while.body.i.i, %entry
  br label %while.body.i.i, !dbg !34

fn3.exit:                                         ; No predecessors!
  %1 = load i32, i32* @f, align 4
  %tobool.i = icmp eq i32 %1, 0
  br label %while.body.i

while.body.i:                                     ; preds = %if.end.i, %fn3.exit
  br i1 %tobool.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %while.body.i
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %while.body.i
  br label %while.body.i
}

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = distinct !DIGlobalVariable(name: "f", scope: !2, file: !3, line: 8, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "PR31381.c", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7)
!7 = distinct !DIGlobalVariable(name: "h", scope: !2, file: !3, line: 8, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"PIC Level", i32 2}
!12 = distinct !DISubprogram(name: "fn4", scope: !3, file: !3, line: 31, type: !13, isLocal: false, isDefinition: true, scopeLine: 32, isOptimized: true, unit: !2, variables: !4)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DILocalVariable(name: "p", arg: 1, scope: !16, file: !3, line: 19, type: !19)
!16 = distinct !DISubprogram(name: "fn2", scope: !3, file: !3, line: 19, type: !17, isLocal: false, isDefinition: true, scopeLine: 20, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !25)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, size: 96, elements: !20)
!20 = !{!21, !23, !24}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !19, file: !3, line: 4, baseType: !22, size: 8, offset: 24)
!22 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !19, file: !3, line: 5, baseType: !8, size: 32, offset: 32)
!24 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !19, file: !3, line: 6, baseType: !8, size: 6, offset: 64, flags: DIFlagBitField, extraData: i64 64)
!25 = !{!15}
!26 = !DIExpression(DW_OP_LLVM_fragment, 72, 24)
!27 = !DILocation(line: 19, column: 20, scope: !16, inlinedAt: !28)
!28 = distinct !DILocation(line: 27, column: 3, scope: !29, inlinedAt: !30)
!29 = distinct !DISubprogram(name: "fn3", scope: !3, file: !3, line: 24, type: !13, isLocal: false, isDefinition: true, scopeLine: 25, isOptimized: true, unit: !2, variables: !4)
!30 = distinct !DILocation(line: 34, column: 7, scope: !31)
!31 = distinct !DILexicalBlock(scope: !12, file: !3, line: 33, column: 5)
!32 = !DIExpression(DW_OP_LLVM_fragment, 0, 24)
!33 = !DILocation(line: 22, column: 9, scope: !16, inlinedAt: !28)
!34 = !DILocation(line: 21, column: 3, scope: !35, inlinedAt: !28)
!35 = !DILexicalBlockFile(scope: !16, file: !3, discriminator: 2)
