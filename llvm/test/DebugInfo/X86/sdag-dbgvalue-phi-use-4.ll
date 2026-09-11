; RUN: llc -start-after=codegenprepare -stop-before finalize-isel -o - %s \
; RUN:    -experimental-debug-variable-locations=false \
; RUN: | FileCheck %s --check-prefixes=CHECK,DBGVALUE
; RUN: llc -start-after=codegenprepare -stop-before finalize-isel -o - %s \
; RUN:    -experimental-debug-variable-locations=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,INSTRREF

; This is a reproducer based on the test case from PR37321.

; We verify that the fragment for the last DBG_VALUE is limited depending
; on the size of the original fragment (and that we do not emit more
; DBG_VALUE instructions than needed in case we cover the whole original
; fragment with just a few DBG_VALUE instructions).

; CHECK-LABEL: bb.{{.*}}.if.end36:
; CHECK:         [[REG1:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 1
; CHECK-NEXT:    [[REG2:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 2
; CHECK-NEXT:    [[REG3:%[0-9]+]]:gr32 = PHI
; INSTRREF-SAME:    debug-instr-number 3
; INSTRREF-NEXT: DBG_INSTR_REF !16, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 0, 32), dbg-instr-ref(1, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !16, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 32, 32), dbg-instr-ref(2, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !16, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 64, 16), dbg-instr-ref(3, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !15, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 10, 32), dbg-instr-ref(1, 0)
; INSTRREF-NEXT: DBG_INSTR_REF !15, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_fragment, 42, 13), dbg-instr-ref(2, 0)
; DBGVALUE-NEXT: DBG_VALUE [[REG1]], $noreg,  !16, !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; DBGVALUE-NEXT: DBG_VALUE [[REG2]], $noreg,  !16, !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; DBGVALUE-NEXT: DBG_VALUE [[REG3]], $noreg,  !16, !DIExpression(DW_OP_LLVM_fragment, 64, 16)
; DBGVALUE-NEXT: DBG_VALUE [[REG1]], $noreg,  !15, !DIExpression(DW_OP_LLVM_fragment, 10, 32)
; DBGVALUE-NEXT: DBG_VALUE [[REG2]], $noreg,  !15, !DIExpression(DW_OP_LLVM_fragment, 42, 13)
; CHECK-NOT:  DBG_

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-w64-windows-gnu"

; Function Attrs: nounwind readnone
define dso_local i64 @nextafterl(i80 %a, i1 %arg) local_unnamed_addr #0 !dbg !5 {
entry:
  br i1 %arg, label %if.else, label %if.then13, !dbg !27

if.then13:                                        ; preds = %entry
  %u.sroa.0.8.insert.insert = or i80 %a, 2222, !dbg !28
  br label %if.end36, !dbg !32

if.else:                                          ; preds = %entry
  br label %if.end36

if.end36:                                         ; preds = %if.else, %if.then13
  %u.sroa.0.1.in = phi i80 [ %u.sroa.0.8.insert.insert, %if.then13 ], [ 1234567, %if.else ]
  call void @llvm.dbg.value(metadata i80 %u.sroa.0.1.in, metadata !16, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 80)), !dbg !33
  call void @llvm.dbg.value(metadata i80 %u.sroa.0.1.in, metadata !15, metadata !DIExpression(DW_OP_LLVM_fragment, 10, 45)), !dbg !33
  %u.sroa.0.0.extract.ashr = ashr i80 %u.sroa.0.1.in, 8, !dbg !34
  %u.sroa.0.0.extract.trunc = trunc i80 %u.sroa.0.0.extract.ashr to i64, !dbg !34
  ret i64 %u.sroa.0.0.extract.trunc
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 330808) (llvm/trunk 330813)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2)
!1 = !DIFile(filename: "pr37321.c", directory: "")
!2 = !{}
!3 = !{i32 1, !"wchar_size", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "nextafterl", scope: !1, file: !1, line: 17, type: !6, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "long double", size: 96, encoding: DW_ATE_float)
!9 = !{!10, !14, !15, !16}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "normal_bit", scope: !5, file: !1, line: 31, type: !12, isLocal: true, isDefinition: true)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!14 = !DILocalVariable(name: "x", arg: 1, scope: !5, file: !1, line: 17, type: !8)
!15 = !DILocalVariable(name: "y", arg: 2, scope: !5, file: !1, line: 17, type: !8)
!16 = !DILocalVariable(name: "u", scope: !5, file: !1, line: 27, type: !24)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "parts", scope: !24, file: !1, line: 26, baseType: !18, size: 128)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !24, file: !1, line: 21, size: 128, elements: !19)
!19 = !{!20, !21, !23}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "mantissa", scope: !18, file: !1, line: 23, baseType: !13, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "expn", scope: !18, file: !1, line: 24, baseType: !22, size: 16, offset: 64)
!22 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "pad", scope: !18, file: !1, line: 25, baseType: !22, size: 16, offset: 80)
!24 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !5, file: !1, line: 19, size: 128, elements: !25)
!25 = !{!26, !17}
!26 = !DIDerivedType(tag: DW_TAG_member, name: "ld", scope: !24, file: !1, line: 20, baseType: !8, size: 96)
!27 = !DILocation(line: 47, column: 7, scope: !5)
!28 = !DILocation(line: 51, column: 14, scope: !29)
!29 = distinct !DILexicalBlock(scope: !30, file: !1, line: 50, column: 11)
!30 = distinct !DILexicalBlock(scope: !31, file: !1, line: 48, column: 5)
!31 = distinct !DILexicalBlock(scope: !5, file: !1, line: 47, column: 7)
!32 = !DILocation(line: 51, column: 2, scope: !29)
!33 = !DILocation(line: 27, column: 5, scope: !5)
!34 = !DILocation(line: 63, column: 22, scope: !35)
!35 = distinct !DILexicalBlock(scope: !5, file: !1, line: 62, column: 7)
