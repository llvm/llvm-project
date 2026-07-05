; RUN: llc < %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

; Verify if dwarfdump contains DBG_VALUE associated with locals.
; See also dgb-value-ti.ll test.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define hidden i32 @fib(i32 %n) local_unnamed_addr #0 !dbg !7 {

entry:
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_variable

  call void @llvm.dbg.value(metadata i32 1, metadata !16, metadata !DIExpression()), !dbg !19
  %cmp8 = icmp sgt i32 %n, 0, !dbg !21
  br i1 %cmp8, label %for.body, label %for.end, !dbg !24

for.body:                                         ; preds = %entry, %for.body
  %b.011 = phi i32 [ %add, %for.body ], [ 1, %entry ]
  %a.010 = phi i32 [ %b.011, %for.body ], [ 0, %entry ]
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]

; CHECK: DW_OP_WASM_location 0x0 0x[[LOCAL_1:[0-9]+]]
  call void @llvm.dbg.value(metadata i32 %b.011, metadata !16, metadata !DIExpression()), !dbg !19

; CHECK-NOT: DW_OP_WASM_location 0x0 0x[[LOCAL_1]]
; CHECK: DW_OP_WASM_location 0x0 0x[[LOCAL_2:[0-9]+]]
  %add = add nsw i32 %b.011, %a.010, !dbg !26
  %inc = add nuw nsw i32 %i.09, 1, !dbg !28
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !19
  %exitcond = icmp eq i32 %inc, %n, !dbg !21
  br i1 %exitcond, label %for.end, label %for.body, !dbg !24, !llvm.loop !29

for.end:                                          ; preds = %for.body, %entry
  %b.0.lcssa = phi i32 [ 1, %entry ], [ %add, %for.body ], !dbg !31
  call void @llvm.dbg.value(metadata i32 %b.0.lcssa, metadata !16, metadata !DIExpression()), !dbg !19
  ret i32 %b.0.lcssa, !dbg !32
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<unknown>", directory: "")
!2 = !{}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "fib", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!16}
!16 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 2, type: !10)
!17 = !DILocation(line: 1, column: 13, scope: !7)
!18 = !DILocation(line: 2, column: 13, scope: !7)
!19 = !DILocation(line: 2, column: 20, scope: !7)
!20 = !DILocation(line: 2, column: 7, scope: !7)
!21 = !DILocation(line: 3, column: 17, scope: !22)
!22 = distinct !DILexicalBlock(scope: !23, file: !1, line: 3, column: 3)
!23 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 3)
!24 = !DILocation(line: 3, column: 3, scope: !23)
!25 = !DILocation(line: 2, column: 10, scope: !7)
!26 = !DILocation(line: 6, column: 7, scope: !27)
!27 = distinct !DILexicalBlock(scope: !22, file: !1, line: 3, column: 27)
!28 = !DILocation(line: 3, column: 23, scope: !22)
!29 = distinct !{!29, !24, !30}
!30 = !DILocation(line: 7, column: 3, scope: !23)
!31 = !DILocation(line: 0, scope: !7)
!32 = !DILocation(line: 8, column: 3, scope: !7)
