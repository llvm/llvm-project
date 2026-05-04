; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s 

define dso_local void @func(i32 noundef %a, ptr noundef %b) !dbg !10 {
; Check the merged debug location of hoisted GEP
; CHECK: entry
; CHECK: %{{[a-zA-Z0-9_]*}} = getelementptr {{.*}} !dbg [[MERGED_DL:![0-9]+]]
; CHECK: [[MERGED_DL]] = !DILocation(line: 0, scope: !{{[0-9]+}})
entry:
  tail call void @llvm.dbg.value(metadata i32 %a, metadata !16, metadata !DIExpression()), !dbg !17
  tail call void @llvm.dbg.value(metadata ptr %b, metadata !18, metadata !DIExpression()), !dbg !17
  %tobool = icmp ne i32 %a, 0, !dbg !19
  br i1 %tobool, label %if.then, label %if.else, !dbg !21

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 1, !dbg !22
  store i32 1, ptr %arrayidx, align 4, !dbg !24
  br label %if.end, !dbg !25

if.else:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds i32, ptr %b, i64 1, !dbg !26
  store i32 1, ptr %arrayidx1, align 4, !dbg !28
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !29
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "main.c", directory: "/root/llvm-test/GVNHoist")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!10 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !14}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!15 = !{}
!16 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !1, line: 1, type: !14)
!19 = !DILocation(line: 2, column: 9, scope: !20)
!20 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 9)
!21 = !DILocation(line: 2, column: 9, scope: !10)
!22 = !DILocation(line: 3, column: 9, scope: !23)
!23 = distinct !DILexicalBlock(scope: !20, file: !1, line: 2, column: 12)
!24 = !DILocation(line: 3, column: 14, scope: !23)
!25 = !DILocation(line: 4, column: 5, scope: !23)
!26 = !DILocation(line: 5, column: 9, scope: !27)
!27 = distinct !DILexicalBlock(scope: !20, file: !1, line: 4, column: 12)
!28 = !DILocation(line: 5, column: 14, scope: !27)
!29 = !DILocation(line: 7, column: 1, scope: !10)
