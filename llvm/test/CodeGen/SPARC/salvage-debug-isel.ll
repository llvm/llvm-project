; RUN: llc -march=sparc -O1 %s -o - -stop-after=finalize-isel | FileCheck %s

; Debug info salvaging in isel means we should see a location for this variable.

; CHECK-LABEL: name:            a
; CHECK: DBG_VALUE %stack.0.b, $noreg, ![[#]], !DIExpression(DW_OP_plus_uconst, 3, DW_OP_stack_value)

define dso_local zeroext i16 @a() local_unnamed_addr #0 !dbg !7 {
entry:
  %b = alloca [6 x i8], align 1
  %arrayidx = getelementptr inbounds [6 x i8], ptr %b, i32 0, i32 undef, !dbg !27
  store i8 4, ptr %arrayidx, align 1, !dbg !28
  %arrayidx1 = getelementptr inbounds i8, ptr %b, i32 3, !dbg !32
    #dbg_value(ptr %arrayidx1, !22, !DIExpression(), !25)
  %0 = load i8, ptr %arrayidx1, align 1, !dbg !33
  %tobool.not = icmp eq i8 %0, 0, !dbg !35
  br i1 %tobool.not, label %if.end, label %for.cond, !dbg !36

for.cond:                                         ; preds = %entry, %for.cond
  br label %for.cond, !dbg !37, !llvm.loop !40

if.end:                                           ; preds = %entry
  ret i16 undef, !dbg !44
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.0.0git.prerel", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "file.c", directory: "/path", checksumkind: CSK_MD5, checksum: "aa7b5139660a2329a6409414c44cc1f6")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!6 = !{!"clang version 20.0.0git.prerel"}
!7 = distinct !DISubprogram(name: "a", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint16_t", file: !11, line: 277, baseType: !12)
!11 = !DIFile(filename: "stdint.h", directory: "", checksumkind: CSK_MD5, checksum: "d9e8f73f3756bbd642f1729623e09484")
!12 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!13 = !{!14, !20, !22}
!14 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 3, type: !15)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 48, elements: !18)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "int8_t", file: !11, line: 298, baseType: !17)
!17 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!18 = !{!19}
!19 = !DISubrange(count: 6)
!20 = !DILocalVariable(name: "c", scope: !7, file: !1, line: 4, type: !21)
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !DILocalVariable(name: "d", scope: !7, file: !1, line: 6, type: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 32)
!25 = !DILocation(line: 0, scope: !7)
!27 = !DILocation(line: 5, column: 3, scope: !7)
!28 = !DILocation(line: 5, column: 8, scope: !7)
!32 = !DILocation(line: 6, column: 16, scope: !7)
!33 = !DILocation(line: 7, column: 33, scope: !34)
!34 = distinct !DILexicalBlock(scope: !7, file: !1, line: 7, column: 7)
!35 = !DILocation(line: 7, column: 7, scope: !34)
!36 = !DILocation(line: 7, column: 7, scope: !7)
!37 = !DILocation(line: 8, column: 5, scope: !38)
!38 = distinct !DILexicalBlock(scope: !39, file: !1, line: 8, column: 5)
!39 = distinct !DILexicalBlock(scope: !34, file: !1, line: 8, column: 5)
!40 = distinct !{!40, !41, !42, !43}
!41 = !DILocation(line: 8, column: 5, scope: !39)
!42 = !DILocation(line: 9, column: 7, scope: !39)
!43 = !{!"llvm.loop.unroll.disable"}
!44 = !DILocation(line: 10, column: 1, scope: !7)
